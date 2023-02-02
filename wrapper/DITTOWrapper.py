import random
from IPython.utils import io
import os
import jsonlines as jsonlines
import pandas as pd
import torch
from ditto.matcher import *
from ditto_light.dataset import DittoDataset
from ditto_light.ditto import evaluate, DittoModel
from ditto_light.exceptions import ModelNotFoundError
from ditto_light.knowledge import *
from ditto_light.summarize import Summarizer
from dotmap import DotMap
from tqdm.autonotebook import tqdm

class DITTOWrapper(object):
    def __init__(self, task, checkpoint_path, exclude_attrs=['id', 'left_id', 'right_id'], one_by_one=False,
                 lm='roberta'):
        self.exclude_attrs = exclude_attrs
        self.one_by_one = one_by_one
        self.checkpoint_path = checkpoint_path

        self.random_n = random.randint(0, 100)

        hp = DotMap()
        hp["task"] = task
        hp["input_path"] = 'test.txt'
        hp["output_path"] = 'output/output_small.jsonl'
        hp["lm"] = lm
        hp["use_gpu"] = True
        hp["fp16"] = True
        hp["checkpoint_path"] = self.checkpoint_path
        hp["dk"] = None
        hp["summarize"] = False
        hp["max_len"] = 64
        self.hp = hp

        config, model = load_model(hp['task'], hp['checkpoint_path'],
                                   hp['lm'], hp['use_gpu'], hp['fp16'])
        self.model = model
        self.config = config
        summarizer = dk_injector = None
        if hp['summarize']:
            summarizer = Summarizer(config, hp.lm)

        if hp['dk'] is not None:
            if 'product' in hp.dk:
                dk_injector = ProductDKInjector(config, hp['dk'])
            else:
                dk_injector = GeneralDKInjector(config, hp['dk'])

        self.summarizer = summarizer
        self.dk_injector = dk_injector
        # tune threshold
        self.threshold = tune_threshold(config, model, hp)

    @staticmethod
    def ditto_format_to_tsv(df, excluded_cols=['id', 'left_id', 'right_id']):
        result_df = pd.DataFrame()
        for prefix in ['left', 'right']:
            mask = ~(df.columns.isin(excluded_cols)) & df.columns.str.startswith(prefix)
            turn_columns = df.columns[mask]
            result_df[prefix] = [''] * df.shape[0]
            for turn_col in turn_columns:
                result_df[prefix] += 'COL ' + turn_col[len(prefix) + 1:] + ' VAL ' + df[turn_col].astype(str) + ' '
        if 'label' in df.columns:
            result_df['label'] = df['label']

        return result_df

    def predict(self, dataset, batch_size=1024):
        """
        Args:
            dataset: dataset to be predicted wih the same structure of the training dataset

        Returns: list of match scores
        """
        self.dataset = dataset
        dataset = dataset.copy().reset_index(drop=True)
        with io.capture_output() as captured:
            self.predictions = []
            file_path = f'/tmp/candidate{self.random_n}.tsv'
            ditto_df = DITTOWrapper.ditto_format_to_tsv(dataset)
            hp = self.hp
            hp["input_path"] = file_path
            batch_size = batch_size

            # ditto_df.to_csv(file_path, sep='\t', header=False, index=False)

            # input_path = hp.input_path
            # if '.jsonl' not in input_path:
            #     with jsonlines.open(input_path + '.jsonl', mode='w') as writer:
            #         for line in open(input_path):
            #             writer.write(line.split('\t')[:2])
            #     input_path += '.jsonl'

            # with jsonlines.open(input_path) as reader:
            #     pairs = []
            #     rows = []
            #     for idx, row in tqdm(enumerate(reader)):
            #         pairs.append(to_str(row[0], row[1], self.summarizer, hp.max_len, self.dk_injector))
            #         rows.append(row)
            pairs = []
            rows = []
            for idx, row in tqdm(enumerate(ditto_df.iloc[:, [0, 1]].values)):
                pairs.append(to_str(row[0], row[1], self.summarizer, hp.max_len, self.dk_injector))
                rows.append(row)

            def chunks(lst, n):
                """Yield successive n-sized chunks from lst."""
                for i in range(0, len(lst), n):
                    yield lst[i:i + n]
                # [lst[i:i + n] for i in range(0, len(lst), n)]

            pred_list, logit_list = [], []
            for pair_chunk in chunks(pairs, batch_size):
                predictions, logits = classify(pair_chunk, self.model, lm=hp.lm, max_len=hp.max_len,
                                               threshold=self.threshold)
                pred_list.append(predictions)
                logit_list.append(logits)
            logit_tensor = torch.cat([torch.tensor(x) for x in logit_list])
            pred_proba = torch.sigmoid(torch.tensor(logit_tensor)[:, 1])
        return pred_proba