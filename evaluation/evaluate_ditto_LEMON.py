
import sys, os

prefix = ''
if '/home/' in os.path.expanduser('~'):  # UNI env
    prefix = '/home/baraldian'
softlab_path = os.path.join(prefix + '/content/drive/Shareddrives/SoftLab/')
project_path = os.path.join(softlab_path, 'Projects', 'WYM')
sys.path.append(os.path.join(project_path, 'notebooks'))
from notebook_import_utility_env import *

import itertools
from Landmark_github.evaluation.Evaluate_explanation_Batch import Evaluate_explanation, calculate_save_metric
from Landmark_github.evaluation.dataset_names import conf_code_map
from copy import deepcopy
import pandas as pd
from Landmark_github.landmark.landmark import Mapper
import os
import lemon
from tqdm.autonotebook import tqdm
from BERT.run_experiments.dataset_names import sorted_dataset_names
from Landmark_github.evaluation.evaluate_ditto import tasks


def convert_to_lemon_dataset(df, prefix_list=['left_', 'right_']):
    columns = df.columns
    side_df_dict = {}
    id_pair_list = []
    id_df = df[['id'] + list(pd.Series(prefix_list) + 'id')]
    id_df = id_df.rename(columns={'id': 'pid', })
    for side, lemon_prefix in zip(prefix_list, ['a.rid', 'b.rid']):
        id_df = id_df.rename(columns={side + 'id': lemon_prefix})
        tdf = df[columns[columns.str.startswith(side)]]
        tdf.columns = tdf.columns.str.replace(side, '')
        tdf = tdf.rename(columns={'id': '__id'})
        tdf = tdf.set_index('__id')
        side_df_dict[side] = tdf
    return list(side_df_dict.values()) + [id_df]


def predict_wrapper_lemon(predict_proba):
    def predict(records_a: pd.DataFrame, records_b: pd.DataFrame, record_id_pairs: pd.DataFrame):
        records_a.columns = 'left_' + records_a.columns
        records_b.columns = 'right_' + records_b.columns

        return predict_proba(records_a.join(records_b))

    return predict


def convert_explanation_lemon_to_df(sample_df, exp):
    columns = np.setdiff1d(sample_df.columns, exclude_attrs + ['label'])
    mapper = Mapper(columns, r' ')

    # k = mapper_fixed.decode_words_to_attr(_)
    # words_with_prefix = .split(mapper_fixed.encode_attr(sample_df))
    splitter = re.compile(' ')
    attr_to_code_map = {v: k for k, v in mapper.attr_map.items()}
    impacts_list = []
    dict_impact = {}
    for key, t_exp in exp.items():
        t_attributions = t_exp.attributions
        words_with_prefix = splitter.split(mapper.encode_attr(sample_df[sample_df['id'] == key]))
        prefix_to_word_map = {x[:3]: x[4:] for x in words_with_prefix}
        dict_impact.update(id=key)
        for t_attr in t_attributions:
            t_position = t_attr.positions[0]
            t_prefix = 'left_' if t_position[0] == 'a' else 'right_'
            t_column = f'{t_prefix}{t_position[1]}'
            pos = t_position[3] if t_position[3] is not None else 0
            t_code = f'{attr_to_code_map[t_column]}{pos:02d}'
            word = prefix_to_word_map[t_code]
            dict_impact.update(column=t_column, position=t_position[3], word=word, word_prefix=f'{t_code}_{word}',
                               impact=t_attr.weight, potential=t_attr.potential)
            impacts_list.append(dict_impact.copy())
    # [x for x in exp[55].attributions if x.positions[0][0]=='a' and x.positions[0][1] == 'Style']
    # pd.DataFrame(impacts_list).sort_values(['id','column','position'])
    return pd.DataFrame(impacts_list)


def compute_explanations_lemon(mirror_elements=False, redo=False, num_explanations=100, evaluate=False, num_round=10):
    os.chdir(os.path.join(softlab_path, 'Projects/external_github/ditto'))

    from wrapper.DITTOWrapper import DITTOWrapper

    checkpoint_path = os.path.join(
        os.path.join(softlab_path, 'Projects/external_github/ditto/checkpoints'))  # 'checkpoints'

    all_data_dict = {}
    batch_size = 2048
    num_samples = 2048  # 2048
    for i in tqdm(range(len(sorted_dataset_names))):

        task = tasks[i]
        turn_dataset_name = sorted_dataset_names[i]
        print('v' * 100)
        print(f'\n\n\n{task: >50}\n' + f'{turn_dataset_name: >50}\n\n\n')
        print('^' * 100)
        turn_dataset_path = os.path.join(dataset_path, turn_dataset_name)
        turn_files_path = os.path.join(base_files_path, turn_dataset_name)
        try:
            os.mkdir(turn_files_path)
        except:
            pass

        dataset_dict = {name: pd.read_csv(os.path.join(turn_dataset_path, f'{name}_merged.csv')) for name in
                        ['train', 'valid', 'test']}
        turn_data_dict = deepcopy(dataset_dict)

        test_df = dataset_dict['test']
        turn_df = test_df
        pos_mask = turn_df['label'] == 1
        pos_df = turn_df[pos_mask]
        neg_df = turn_df[~pos_mask]
        pos_sample = pos_df.sample(num_explanations, random_state=0) if pos_df.shape[0] >= num_explanations else pos_df
        neg_sample = neg_df.sample(num_explanations, random_state=0) if neg_df.shape[0] >= num_explanations else neg_df
        turn_data_dict['pos_sample'] = pos_sample
        turn_data_dict['neg_sample'] = neg_sample

        model = None
        conf_name = 'LEMON'
        try:
            assert redo is False, 'redo was set.'
            for sample, prefix in zip([pos_sample, neg_sample], ['positive', 'negative']):
                name = f'{prefix}_explanations_{conf_name}'
                tmp_path = os.path.join(turn_files_path, f'{name}.csv')
                res_df = pd.read_csv(tmp_path)
                turn_data_dict[name] = res_df
                assert res_df.id.nunique() >= sample.shape[0], 'Not enough sample computed'
                print('loaded')
        except Exception as e:
            print(e)
            model = DITTOWrapper(task, checkpoint_path)
            ditto_pred_proba = model.predict
            pred_proba_lemon = predict_wrapper_lemon(ditto_pred_proba)
            for sample, prefix in zip([pos_sample, neg_sample], ['positive', 'negative']):
                a, b, id_df = convert_to_lemon_dataset(sample)
                exp = lemon.explain(a, b, id_df, pred_proba_lemon,
                                    granularity='tokens',
                                    num_features=200,
                                    token_representation="independent",
                                    )
                exp_df = convert_explanation_lemon_to_df(sample, exp)
                name = f'{prefix}_explanations_{conf_name}'
                tmp_path = os.path.join(turn_files_path, f'{name}.csv')
                exp_df.to_csv(tmp_path, index=False)
                tmp_path = os.path.join(turn_files_path, f'{prefix}_explanations_{conf_name}_raw.pickle')
                with open(tmp_path, 'wb') as file:
                    pickle.dump(exp, file)
                turn_data_dict[name] = exp_df
        all_data_dict[turn_dataset_name] = deepcopy(turn_data_dict)

        if evaluate is True:
            if model is None:
                model = DITTOWrapper(task, checkpoint_path)
            ditto_pred_proba = model.predict
            to_iter = itertools.product(zip([pos_sample, neg_sample], ['positive', 'negative']), ['impacts_accuracy', 'utility'])
            evaluation_res = {}
            for (sample, prefix), eval_name in to_iter:
                utility = eval_name == 'utility'
                name = f'{prefix}_explanations_{conf_name}'
                exp_df = turn_data_dict[name]
                exp_df['conf'] = conf_name
                ids = sample['id'].unique()
                ev = Evaluate_explanation(sample, exp_df, percentage=.25, num_round=num_round,
                                          predict_method=ditto_pred_proba, exclude_attrs=exclude_attrs)
                ev.variable_cols = np.setdiff1d(sample.columns, exclude_attrs + ['label'])


                res_df = ev.evaluate_set(ids, conf_name, variable_side='all', utility='all')

                res_df['conf_code'] = res_df.conf.map(conf_code_map)
                res = calculate_save_metric(res_df=res_df, model_files_path=turn_files_path, metric_name='all', prefix=name)
                # res, _ = aggregate_results(res_df, model_files_path=turn_files_path, utility=utility, prefix=name)
                # tmp_path = os.path.join(turn_files_path, f'{prefix}_{eval_name}_{conf_name}.csv')
                # res.to_csv(tmp_path, index=False)
                tmp_path = os.path.join(turn_files_path, f'{prefix}_{eval_name}_{conf_name}_data.csv')
                # check concorde column and same_class --> OK
                res_df.to_csv(tmp_path, index=False)
                print(tmp_path + '-- done')


    return all_data_dict

#
# def evaluate_ditto_lemon(num_explanations = 100 ):
#     from functools import partial
#
#     checkpoint_path = os.path.join(
#         os.path.join(softlab_path, 'Projects/external_github/ditto/checkpoints'))  # 'checkpoints'
#
#     batch_size = 2048
#     num_samples = 2048
#     for i in tqdm(range(len(sorted_dataset_names))):
#
#         task = tasks[i]
#         turn_dataset_name = sorted_dataset_names[i]
#         print('v' * 100)
#         print(f'\n\n\n{task: >50}\n' +
#               f'{turn_dataset_name: >50}\n\n\n')
#         print('^' * 100)
#         turn_dataset_path = os.path.join(dataset_path, turn_dataset_name)
#         turn_files_path = os.path.join(base_files_path, turn_dataset_name)
#         try:
#             os.mkdir(turn_files_path)
#         except:
#             pass
#
#         dataset_dict = {name: pd.read_csv(os.path.join(turn_dataset_path, f'{name}_merged.csv')) for name in
#                         ['train', 'valid', 'test']}
#
#         # model = DITTOWrapper(task, checkpoint_path)
#         # explainer = Landmark(partial(model.predict, batch_size=batch_size), test_df, exclude_attrs=excluded_cols + ['label','id'], lprefix='left_', rprefix='right_', split_expression=r' ')
#         test_df = dataset_dict['test']
#         turn_df = test_df
#         pos_mask = turn_df['label'] == 1
#         pos_df = turn_df[pos_mask]
#         neg_df = turn_df[~pos_mask]
#         pos_sample = pos_df.sample(num_explanations, random_state=0) if pos_df.shape[0] >= num_explanations else pos_df
#         neg_sample = neg_df.sample(num_explanations, random_state=0) if neg_df.shape[0] >= num_explanations else neg_df
#
#         exp_dict = {}
#         for sample, prefix in zip([pos_sample, neg_sample], ['positive', 'negative']):
#             exp_dict[prefix] = []
#             conf_list = ['single', 'double', 'LIME']
#             if prefix == 'negative':
#                 conf_list.append('MOJITO')
#             for conf in conf_list:
#                 tmp_path = os.path.join(turn_files_path, f'{prefix}_explanations_{conf}.csv')
#                 tmp_df = pd.read_csv(tmp_path)
#                 assert tmp_df.id.nunique() >= sample.shape[0], 'Not computed'
#                 print('loaded')
#                 tmp_df['conf'] = conf
#                 exp_dict[prefix].append(tmp_df)
#
#             total_exp = pd.concat(exp_dict[prefix]).drop(['index'], 1)
#             if prefix == 'negative':
#                 total_exp = total_exp.drop(['exp', 'tuple'], 1)
#             total_exp['variable_side'] = np.where(total_exp['column'].str.startswith('left'), 'left', 'right')
#
#             def map_conf(x):
#                 if x['conf'] == 'LIME':
#                     return 'LIME'
#                 elif x['conf'] == 'MOJITO':
#                     return 'mojito_copy_' + ('R' if x['variable_side'] == 'right' else 'L')
#                 else:
#                     return x['variable_side'] + ('Copy' if x['conf'] == 'double' else '')
#
#             total_exp['conf'] = total_exp.apply(map_conf, 1)
#             exp_dict[prefix] = total_exp
#
#         model = DITTOWrapper(task, checkpoint_path)
#         explainer = Landmark(
#             partial(model.predict, batch_size=batch_size),
#             # lambda x: [0.5] * x.shape[0],
#             test_df, exclude_attrs=excluded_cols + ['label', 'id'],
#             lprefix='left_', rprefix='right_', split_expression=r' ')
#
#         # for prefix in ['negative']:
#         #     for suffix in ['utility']:
#
#         for prefix in ['positive', 'negative']:
#             for suffix in ['impacts_accuracy', 'utility']:
#                 if prefix == 'positive':
#                     f = partial(evaluate_explanation_positive, utility=suffix == 'utility')
#                 elif prefix == 'negative':
#                     f = partial(evaluate_explanation_negative, utility=suffix == 'utility')
#                 res, data_test = f(exp_dict[prefix], explainer)
#                 tmp_path = os.path.join(turn_files_path, f'{prefix}_{suffix}.csv')
#                 res.to_csv(tmp_path, index=False)
#                 print(tmp_path + '-- done')
#                 tmp_path = os.path.join(turn_files_path, f'{prefix}_{suffix}_data.csv')
#                 data_test.to_csv(tmp_path, index=False)


if __name__ == "__main__":
    compute_explanations_lemon(num_explanations=100, evaluate=True, )
