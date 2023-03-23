import gc
from functools import partial
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from tqdm.notebook import tqdm

from ..landmark.landmark import Landmark, PlotExplanation

from .removal_strategies import get_tokens_to_remove_five, get_tokens_to_remove_incrementally, \
    get_tokens_to_remove_aopc, get_tokens_to_remove_degradation, get_tokens_to_remove_sufficiency, \
    get_tokens_to_remove_single_units, get_tokens_to_change_class


"""def _evaluate_df(word_relevance, df_to_process, predictor, exclude_attrs=('id', 'left_id', 'right_id', 'label'),
                score_col='pred'):
    print(f'Testing unit remotion with -- {score_col}')
    assert df_to_process.shape[
               0] > 0, f'DataFrame to evaluate must have some elements. Passed df has shape {df_to_process.shape[0]}'
    evaluation_df = df_to_process.copy().replace(pd.NA, '')
    word_relevance_prefix = append_prefix(evaluation_df, word_relevance)

    if score_col == 'pred':
        word_relevance_prefix['impact'] = word_relevance_prefix[score_col] - 0.5
    else:
        word_relevance_prefix['impact'] = word_relevance_prefix[score_col]

    word_relevance_prefix['conf'] = 'bert'

    res_list = list()

    for side in ['left', 'right']:
        evaluation_df['pred'] = predictor(evaluation_df)
        side_word_relevance_prefix = word_relevance_prefix.copy()
        side_word_relevance_prefix['word_prefix'] = side_word_relevance_prefix[side + '_word_prefixes']
        side_word_relevance_prefix = side_word_relevance_prefix.query(f'{side}_word != "[UNP]"')
        ev = EvaluateExplanation(side_word_relevance_prefix, evaluation_df, predict_method=predictor,
                                 exclude_attrs=exclude_attrs, percentage=.25, num_rounds=3)

        fixed_side = 'right' if side == 'left' else 'left'
        res_df = ev.evaluate_set(df_to_process.id.values, 'bert', variable_side=side, fixed_side=fixed_side,
                                 utility=True)
        res_list.append(res_df.copy())

    return pd.concat(res_list)"""


def correlation_vs_landmark(df, word_relevance, predictor, match_ids, no_match_ids, score_col='pred', num_samples=250):
    """
    test code
    from Evaluation import correlation_vs_landmark
    df = routine.valid_merged
    word_relevance = routine.words_pairs_dict['valid']
    match_ids, no_match_ids = [10],[15]
    predictor = routine.get_predictor()
    correlation_data = correlation_vs_landmark(df, word_relevance, predictor, match_ids,
                                                                       no_match_ids)
    """
    print(f'Testing Landmark correlation with -- {score_col}')
    explainer = Landmark(predictor, df, exclude_attrs=('id', 'label'), lprefix='left_', rprefix='right_')
    res_list_of_dict = []
    for match_code, id_samples in zip(['match', 'nomatch'], [match_ids, no_match_ids]):
        res_dict = {'match_code': match_code}
        print(f'Evaluating {match_code}')
        for id in tqdm(id_samples):
            word_relevance_sample = word_relevance[word_relevance.id == id]
            df_sample = df[df.id == id]
            # display(df_sample)
            res_dict.update(id=id)
            exp = explainer.explain(df_sample, num_samples=num_samples, conf='single')
            for side, landmark_side in zip(['left', 'right'], ['right', 'left']):
                # print(f'side:{side} -- landmark:{landmark_side}')
                res_dict.update(side=side)
                # display(exp)
                landmark_impacts = exp.query(f'conf =="{landmark_side}_landmark"')
                landmark_impacts[side + '_attribute'] = landmark_impacts['column'].str[len(side + '_'):]
                landmark_impacts[side + '_word'] = landmark_impacts['word']
                landmark_impacts = landmark_impacts[[side + '_word', side + '_attribute', 'impact']]
                words_relevance_tmp = word_relevance_sample.query(side + '_attribute != "[UNP]"')[
                    [side + '_word', side + '_attribute', 'id', score_col]]
                words_relevance_tmp['relevance'] = words_relevance_tmp[score_col]
                # display(words_relevance_tmp, landmark_impacts)
                impacts_comparison = words_relevance_tmp.merge(landmark_impacts,
                                                               on=[side + '_attribute', side + '_word'])
                # display(impacts_comparison)
                for method in ['pearson', 'kendall', 'spearman']:
                    corr = impacts_comparison['impact'].corr(impacts_comparison['relevance'], method=method)
                    res_dict[method] = corr
                res_list_of_dict.append(res_dict.copy())

    return pd.DataFrame(res_list_of_dict)


def generate_altered_df(df, y_true, word_relevance_df, tokens_to_remove):
    new_df = df.copy()
    for i in tqdm(range(df.shape[0])):
        el = new_df.iloc[[i]]
        id = el['id'].values[0]
        turn_tokens_to_remove = tokens_to_remove.query(f'id == {id}')
        # wr_el = word_relevance_df.query(f'id == {id}')
        # tokens_to_remove = token_remotion_fn(wr_el)
        for side in ['left', 'right']:
            tokens_to_remove_side = turn_tokens_to_remove[[side + '_word', side + '_attribute']].values
            for word, attr in tokens_to_remove_side:
                if word != '[UNP]':
                    try:
                        el[side + '_' + attr] = el[side + '_' + attr].str.replace(word, '', regex=False).str.strip()
                    except Exception as e:
                        print(e)
                        display(el, side + '_' + attr, word)
                        assert False
        if (el[np.setdiff1d(new_df.columns, ['id', 'left_id', 'label', 'right_id'])] != '').any(1).values[0]:
            new_df.iloc[[i]] = el

    return new_df


def process_roc_auc(y_true, y_pred, plot=True):
    # display(df.iloc[[10]], new_df.iloc[[10]])
    if plot:
        fpr, tpr, thresholds = roc_curve(y_true.astype(int), y_pred)

        fig = px.area(
            x=fpr, y=tpr,
            title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
            width=700, height=500
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )

        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_xaxes(constrain='domain')
        fig.show()
    auc_score = roc_auc_score(y_true.astype(int), y_pred)

    return auc_score


def token_removal_delta_performance(df, y_true, word_relevance, predictor, k_list=(10, 5, 3, 1), plot=True,
                                    score_col='pred'):
    print(f'Testing {score_col}!')
    tokens_dict = {}
    if score_col == 'pred':
        th = 0.5
    else:
        th = 0

    x = word_relevance.copy()
    x['tmp_score'] = x[score_col] * np.where(x.label.values == 1, 1, -1)
    tokens_dict['del_useful'] = x[(x[score_col] >= th) == (x.label.values == 1)].sort_values('tmp_score',
                                                                                             ascending=False).groupby(
        'id').head
    # .sort_values(score_col, ascending=(x.label.values[0] == 0))
    tokens_dict['del_useless'] = x[(x[score_col] < th) == (x.label.values == 1)].sort_values('tmp_score',
                                                                                             ascending=True).groupby(
        'id').head
    tokens_dict['del_random'] = partial(x.groupby('id').sample, random_state=0)
    # ascending (low is useful) for nomatch
    # .sort_values(score_col, ascending=(x.label.values[0] == 1))
    # ascending (low is useless (NOT useful)) for match

    res_list = []
    tmp_dict = {}

    print('Evaluating delta performance on full dataset with token remotion.')
    for k in k_list:
        tmp_dict.update(n_tokens=k)
        df_dict = {}
        gc.collect()
        torch.cuda.empty_cache()
        for fn_name in ['del_random', 'del_useful', 'del_useless']:
            code = f'{fn_name}-{k}'
            print(code)
            if fn_name != 'del_random':
                tokens_to_remove = tokens_dict[fn_name](k)
                turn_df = df
                turn_word_relevance = word_relevance
            else:
                sample_mask = x.groupby('id')[score_col].count() >= k
                turn_df = df.sort_values('id')[sample_mask]
                turn_word_relevance = word_relevance[word_relevance.id.isin(turn_df['id'].values)]
                tokens_to_remove = turn_word_relevance.groupby('id').sample(k, random_state=0)
            altered_df = generate_altered_df(turn_df, turn_df.label.values.astype(int), turn_word_relevance,
                                             tokens_to_remove=tokens_to_remove)
            df_dict[code] = altered_df

        pred_dict = {}
        all_df = pd.concat([value for key, value in df_dict.items()])
        all_df['id'] = np.arange(all_df.shape[0])
        print('Predicting')
        all_pred = predictor(all_df)
        start = 0
        for key, value in df_dict.items():
            stop = value.shape[0] + start
            pred_dict[key] = all_pred[start:stop]
            start = stop
        for fn_name in ['del_random', 'del_useful', 'del_useless']:
            tmp_dict['function'] = fn_name
            code = f'{fn_name}-{k}'
            new_pred = pred_dict[code]
            print(code)
            turn_y_true = df_dict[code].label.values.astype(int)
            auc_score = process_roc_auc(turn_y_true, new_pred, plot)
            tmp_dict['auc_score'] = auc_score
            for score_name, scorer in [['f1', f1_score], ['precision', precision_score], ['recall', recall_score]]:
                tmp_dict[score_name] = scorer(turn_y_true, new_pred > .5)
            res_list.append(tmp_dict.copy())
            pred_dict[code] = new_pred

    return pd.DataFrame(res_list)


class EvaluateExplanation(Landmark):

    def __init__(self, dataset: pd.DataFrame, impacts_df: pd.DataFrame, percentage: float=.25, num_rounds: int=10,
                 evaluate_removing_du: bool=False, recompute_embeddings: bool=True, variable_side: str='all',
                 fixed_side: str='all', add_before_perturbation=None, add_after_perturbation=None, **kwargs):
        """
        Args:
            evaluate_removing_du: specific parameter for WYM: evaluate Decision Units impacts instead of removing
            single tokens.
            recompute_embeddings: specific parameter for WYM: recompute embeddings after removal of Decision Units or
            single tokens. This parameter must be True if evaluate_removing_du is False.
        """

        if not recompute_embeddings and not evaluate_removing_du:
            ValueError("It is not possible to evaluate on single tokens without recomputing embeddings.")

        super().__init__(dataset=dataset, variable_side=variable_side, fixed_side=fixed_side,
                         add_before_perturbation=add_before_perturbation, add_after_perturbation=add_after_perturbation,
                         **kwargs)

        # init of evaluation parameters
        self.impacts_df = impacts_df

        if 'label' in self.impacts_df.columns and (self.impacts_df['label'] == 1).all():
            self.positive_examples = True
        else:
            self.positive_examples = False

        self.percentage = percentage
        self.num_rounds = num_rounds

        self.evaluate_removing_du = evaluate_removing_du  # was decision_unit_view, default False
        self.recompute_embeddings = recompute_embeddings  # was remove_decision_unit_only, default False

        self.add_after_perturbation = add_after_perturbation
        self.add_before_perturbation = add_before_perturbation

        self.fixed_accepted_sides = frozenset(['left', 'right', 'all', ''])
        self.variable_accepted_sides = frozenset(['left', 'right', 'all'])

        if variable_side not in self.variable_accepted_sides:
            raise ValueError("Invalid settings: variable side is not left, right or all.")

        if fixed_side not in self.fixed_accepted_sides:
            raise ValueError("Invalid settings: fixed side is not left, right or all.")

        if variable_side in ('left', 'right') and fixed_side == 'all':
            raise ValueError(f"Invalid settings: variable side is {variable_side} but fixed side is {fixed_side}.")

        if fixed_side in ('left', 'right') and variable_side == 'all':
            raise ValueError(f"Invalid settings: fixed side is {fixed_side} but variable side is {variable_side}.")

        if variable_side == 'all' and fixed_side == 'all':
            fixed_side = str()

        if variable_side == fixed_side:
            raise ValueError(f"Invalid settings: variable and fixed sides are the same "
                             f"({variable_side}, {fixed_side}).")

        self.variable_side = variable_side
        self.fixed_side = fixed_side

        self.impacts = list()
        self.words_with_prefixes = list()
        self.variable_encoded = list()
        self.fixed_data_list = list()
        self.fixed_data_df = None
        self.start_pred = None
        self.data_list = list()
        self.preds = None
        self.old_res_df = None

        # defined at runtime
        self.res_df = None
        self.counterfactual_examples = None
        self.best_counterfactuals = None
        self.counterfactuals_plotting_data = None

    def update_settings(self, **kwargs) -> None:
        """
        Method to update the variable and fixed sides for the evaluation.
        Args:
            **kwargs: EvaluateExplanation and Landmark parameters to update.

        Returns:
            None
        """

        super().update_settings(**kwargs)

        if 'impacts_df' in kwargs:
            self.impacts_df = kwargs['impacts_df']

            if 'label' in self.impacts_df.columns and (self.impacts_df['label'] == 1).all():
                self.positive_examples = True
            else:
                self.positive_examples = False

        if 'percentage' in kwargs:
            self.percentage = kwargs['percentage']

        if 'num_rounds' in kwargs:
            self.num_rounds = kwargs['num_rounds']

        evaluate_removing_du = kwargs['evaluate_removing_du'] \
            if 'evaluate_removing_du' in kwargs else self.evaluate_removing_du
        recompute_embeddings = kwargs['recompute_embeddings'] \
            if 'recompute_embeddings' in kwargs else self.recompute_embeddings

        if not recompute_embeddings and not evaluate_removing_du:
            ValueError("Invalid settings: it is not possible to evaluate on single tokens without "
                       "recomputing embeddings.")
        else:
            self.evaluate_removing_du = evaluate_removing_du
            self.recompute_embeddings = recompute_embeddings

        variable_side = kwargs['variable_side'] if 'variable_side' in kwargs else str()
        fixed_side = kwargs['fixed_side'] if 'fixed_side' in kwargs else str()

        if variable_side or fixed_side:
            if not variable_side and fixed_side:
                raise ValueError(f"Invalid settings: variable side is empty but fixed side is {fixed_side}.")

            if variable_side:
                if variable_side not in self.variable_accepted_sides:
                    raise ValueError("Invalid settings: variable side is not left, right or all.")

                if (variable_side in ('left', 'right') and fixed_side == 'all') or \
                        (variable_side in ('left', 'right') and not fixed_side):
                    raise ValueError(
                        f"Invalid settings: variable side is {variable_side} but fixed side is empty.")

            if fixed_side:
                if fixed_side not in self.fixed_accepted_sides:
                    raise ValueError("Invalid settings: fixed side is not left, right or all.")

                if (fixed_side in ('left', 'right') and variable_side == 'all') or \
                        (fixed_side in ('left', 'right') and not variable_side):
                    raise ValueError(
                        f"Invalid settings: fixed side is {fixed_side} but variable side is empty.")

            if variable_side == 'all' and fixed_side:
                print(f"Warning, invalid settings: variable side is {variable_side} and fixed side is not empty. "
                      f"Ignoring fixed side value.")
                fixed_side = str()

            if variable_side == fixed_side:
                raise ValueError(f"Invalid settings: variable and fixed sides are the same "
                                 f"({variable_side}, {fixed_side}).")

            if variable_side:
                self.variable_side = variable_side
                self.fixed_side = fixed_side

        self.impacts.clear()
        self.words_with_prefixes.clear()
        self.variable_encoded.clear()
        self.fixed_data_list.clear()
        self.fixed_data = None
        self.fixed_data_df = None
        self.start_pred = None
        self.data_list.clear()
        self.preds = None
        self.old_res_df = self.res_df
        self.res_df = None
        self.counterfactual_examples = None
        self.best_counterfactuals = None
        self.counterfactuals_plotting_data = None

        print("EvaluateExplanation settings updated.")

    def plot_counterfactual(self, pred_percentage: bool=True, palette: list=seaborn.color_palette().as_hex(),
                            spaced_attributes: bool=True, beautify_table: bool=True, align_left: bool=False):
        return PlotExplanation.plot_counterfactual(self.counterfactuals_plotting_data, pred_percentage, palette,
                                                   positive_examples=self.positive_examples,
                                                   spaced_attributes=spaced_attributes, beautify_table=beautify_table,
                                                   align_left=align_left)

    def generate_counterfactual_examples(self):
        def generate_description_from_side_attributes(df_row, side='left'):
            if side not in ('left', 'right'):
                raise ValueError("Wrong side value")

            return ' '.join(
                [str(df_row[key]) for key in df_row.keys() if
                 key.startswith(f'{side}_') and
                 df_row[key] is not np.nan and
                 key not in (f'{side}_id', f'{side}_description')])

        def split_left_and_right_word_prefixes(group_df):
            encoded_left_desc = group_df[group_df['column'].str.startswith('left_')]['word_prefix'].to_list()
            encoded_right_desc = group_df[group_df['column'].str.startswith('right_')]['word_prefix'].to_list()
            return pd.Series([[encoded_left_desc, encoded_right_desc]], index=['encoded_descs'])

        counterfactual_examples = self.res_df[self.res_df['detected_delta'] >= 0.5]
        best_counterfactuals = counterfactual_examples.loc[
            counterfactual_examples.groupby(['id'])['num_tokens'].idxmax()]

        landmark_plotting_data = best_counterfactuals.merge(self.dataset, on='id')

        landmark_plotting_data['left_description'] = landmark_plotting_data.apply(
            generate_description_from_side_attributes, axis=1)
        landmark_plotting_data['right_description'] = landmark_plotting_data.apply(
            generate_description_from_side_attributes, side='right',
            axis=1)

        encoded_descriptions = self.variable_mapper.encode_elements(self.dataset).groupby('id').apply(
            split_left_and_right_word_prefixes)

        self.counterfactual_examples = counterfactual_examples
        self.best_counterfactuals = best_counterfactuals
        self.counterfactuals_plotting_data = landmark_plotting_data.merge(encoded_descriptions, on='id')

        return self.counterfactuals_plotting_data

    def prepare_impacts(self):
        for id_ in self.dataset.id.unique():
            turn_variable_encoded = None
            impacts_sorted = self.impacts_df.query(f'id == {id_}')

            if not self.evaluate_removing_du:
                prefix = self.lprefix if self.variable_side == 'left' else self.rprefix
                impacts_sorted = impacts_sorted.query(f'attribute.str.startswith("{prefix}")')

            impacts_sorted = impacts_sorted.sort_values('impact', ascending=False).reset_index(drop=True)
            self.impacts.append(impacts_sorted['impact'].values)

            if self.recompute_embeddings:
                if self.evaluate_removing_du:
                    self.words_with_prefixes.append(
                        [impacts_sorted['left_word_prefixes'].values, impacts_sorted['right_word_prefixes'].values])
                else:
                    self.words_with_prefixes.append(impacts_sorted['word_prefix'].values)

                turn_variable_encoded = self.prepare_element(self.dataset[self.dataset.id == id_].copy())

            else:
                if self.evaluate_removing_du:
                    self.words_with_prefixes.append(impacts_sorted)
                    turn_variable_encoded = impacts_sorted
                    self.fixed_data = None
                # else: instance class state is invalid, managed in init

            self.fixed_data_list.append(self.fixed_data)
            self.variable_encoded.append(turn_variable_encoded)

        if self.fixed_data_list[0] is not None:
            self.fixed_data_df = pd.concat(self.fixed_data_list)
        else:
            self.fixed_data_df = None

        self.start_pred = self.restructure_and_predict(self.variable_encoded)[:, 1]  # match_score

    def restructure_strings(self, perturbed_strings):
        """

        Decode :param perturbed_strings into DataFrame and
        :return reconstructed pairs appending the landmark entity.

        """

        df_list = list()
        for single_row in perturbed_strings:
            df_list.append(self.variable_mapper.decode_words_to_attr_dict(single_row))
        variable_df = pd.DataFrame.from_dict(df_list)
        if self.add_after_perturbation is not None:
            self.add_tokens(variable_df, variable_df.columns, self.add_after_perturbation, overlap=self.overlap)
        if self.fixed_data is not None:
            fixed_df = self.fixed_data_df
            fixed_df.reset_index(inplace=True, drop=True)
        else:
            fixed_df = None
        return pd.concat([variable_df, fixed_df], axis=1)

    def restructure_and_predict(self, perturbed_strings):
        """
            Restructure the perturbed strings from the perturbation system and return the related predictions.
        """
        non_null_rows = np.array([True] * len(perturbed_strings))
        
        if self.recompute_embeddings:
            self.tmp_dataset = self.restructure_strings(perturbed_strings)
            self.tmp_dataset.reset_index(inplace=True, drop=True)
        
        else:
            for i, value in enumerate(perturbed_strings):
                if value.empty:
                    non_null_rows[i] = False
                else:
                    value['id'] = i
            self.tmp_dataset = pd.concat(perturbed_strings)
            
        ret = np.ndarray(shape=(len(perturbed_strings), 2))
        ret[:, :] = 0.5
        predictions = self.model_predict(self.tmp_dataset)
        # assert len(perturbed_strings) == len(predictions), f'df and predictions shape are misaligned'

        ret[non_null_rows, 1] = np.array(predictions)
        ret[:, 0] = 1 - ret[:, 1]
        return ret


    def generate_descriptions(self, combinations_to_remove, words_with_prefixes, variable_encoded):
        description_to_evaluate = list()
        comb_name_sequence = list()
        tokens_to_remove_sequence = list()
        for comb_name, combinations in combinations_to_remove.items():
            for tokens_to_remove in combinations:
                tmp_encoded = variable_encoded
                if self.evaluate_removing_du:  # remove both tokens of left and right as a united view without landmark
                    if not self.recompute_embeddings:
                        tmp_encoded = tmp_encoded.drop(tokens_to_remove)
                    else:
                        for turn_word_with_prefixes in words_with_prefixes:
                            for token_with_prefix in turn_word_with_prefixes[tokens_to_remove]:
                                tmp_encoded = tmp_encoded.replace(str(token_with_prefix), '')
                else:
                    for token_with_prefix in words_with_prefixes[tokens_to_remove]:
                        tmp_encoded = tmp_encoded.replace(str(token_with_prefix), '')
                description_to_evaluate.append(tmp_encoded)
                comb_name_sequence.append(comb_name)
                tokens_to_remove_sequence.append(tokens_to_remove)
        return description_to_evaluate, comb_name_sequence, tokens_to_remove_sequence

    def evaluate_impacts(self, utility=False, k=5):

        self.prepare_impacts()

        data_list = list()
        description_to_evaluate_list = list()
        for index, id_ in enumerate(self.dataset.id.unique()):
            all_comb = dict()
            if utility is False:
                turn_comb = get_tokens_to_remove_five(self.start_pred[index], self.impacts[index],
                                                      self.percentage, self.num_rounds)
                all_comb.update(**turn_comb)
            if utility is True or utility == 'all':
                turn_comb = {
                    'change_class': [
                        get_tokens_to_change_class(self.start_pred[index], self.impacts[index])
                    ],
                    'single_word': [
                        [x] for x in np.arange(self.impacts[index].shape[0])
                    ],
                     'all_opposite': [
                         [pos for pos, impact in enumerate(self.impacts[index]) if
                                               (impact > 0) == (self.start_pred[index] > .5)]
                     ],
                    'change_class_D.10': [
                        get_tokens_to_change_class(self.start_pred[index], self.impacts[index], delta=.1)
                    ],
                    'change_class_D.15': [
                        get_tokens_to_change_class(self.start_pred[index], self.impacts[index], delta=.15)
                    ]
                }

                all_comb.update(**turn_comb)

            if utility == 'AOPC' or utility == 'all':
                turn_comb = get_tokens_to_remove_aopc(self.impacts[index], self.num_rounds, k=k)
                all_comb.update(**turn_comb)

            if utility == 'sufficiency' or utility == 'all':
                turn_comb = get_tokens_to_remove_sufficiency(self.start_pred[index], self.impacts[index],
                                                             self.num_rounds, k=k)
                all_comb.update(**turn_comb)

            if utility == 'degradation' or utility == 'all':
                turn_comb = get_tokens_to_remove_degradation(self.start_pred[index], self.impacts[index],
                                                             self.num_rounds)
                all_comb.update(**turn_comb)

            if utility == 'single_units' or utility == 'all':
                turn_comb = get_tokens_to_remove_single_units(self.impacts[index])
                all_comb.update(**turn_comb)

            if utility == 'incremental' or utility == 'all':
                turn_comb = get_tokens_to_remove_incrementally(self.impacts[index], limit=k)
                all_comb.update(**turn_comb)

            description_to_evaluate, comb_name_sequence, tokens_to_remove_sequence = self.generate_descriptions(
                all_comb, self.words_with_prefixes[index], self.variable_encoded[index])
            data_list.append([description_to_evaluate, comb_name_sequence, tokens_to_remove_sequence])
            self.data_list = data_list
            description_to_evaluate_list.append(description_to_evaluate)

        if self.fixed_data_list[0] is not None:
            self.fixed_data_df = pd.concat(
                [self.fixed_data_list[i] for i, x in enumerate(description_to_evaluate_list) for _ in range(len(x))])
        else:
            self.fixed_data_df = None
        all_descriptions = np.concatenate(description_to_evaluate_list)
        preds = self.restructure_and_predict(all_descriptions)[:, 1]
        assert len(preds) == len(all_descriptions)
        splitted_preds = list()
        start_idx = 0
        for turn_desc in description_to_evaluate_list:
            end_idx = start_idx + len(turn_desc)
            splitted_preds.append(preds[start_idx: end_idx])
            start_idx = end_idx
        self.preds = preds
        res_list = list()
        for index, id_ in enumerate(self.dataset.id.unique()):
            evaluation = {'id': id_, 'start_pred': self.start_pred[index]}
            desc, comb_name_sequence, tokens_to_remove_sequence = data_list[index]
            impacts = self.impacts[index]
            start_pred = self.start_pred[index]
            words_with_prefixes = self.words_with_prefixes[index]
            for new_pred, tokens_to_remove, comb_name in zip(splitted_preds[index], tokens_to_remove_sequence,
                                                             comb_name_sequence):
                correct = (new_pred > .5) == ((start_pred - np.sum(impacts[tokens_to_remove])) > .5)
                evaluation.update(comb_name=comb_name, new_pred=new_pred, correct=correct,
                                  expected_delta=np.sum(impacts[tokens_to_remove]),
                                  detected_delta=-(new_pred - start_pred),
                                  num_tokens=impacts.shape[0]
                                  )

                if self.evaluate_removing_du:
                    if not self.recompute_embeddings:
                        evaluation.update(tokens_removed=words_with_prefixes.loc[
                            tokens_to_remove, ['left_word_prefixes', 'right_word_prefixes']].values.tolist())
                    else:
                        evaluation.update(tokens_removed=list(
                            [list(turn_pref[tokens_to_remove]) for turn_pref in words_with_prefixes]))

                else:
                    # TODO: check this in case of self.evaluate_removing_du = False after update_settings
                    evaluation.update(tokens_removed=list(words_with_prefixes[tokens_to_remove]))
                res_list.append(evaluation.copy())

        self.res_df = pd.DataFrame(res_list)
        self.res_df['error'] = self.res_df.expected_delta - self.res_df.detected_delta

        return self.res_df


    def evaluate_set(self, ids, conf_name, variable_side='left', fixed_side='right', add_before_perturbation=None,
                     add_after_perturbation=None, overlap=True, utility=False):
        impacts_all = self.impacts_df[(self.impacts_df.conf == conf_name)]
        res = []
        if variable_side == 'all':
            impacts_all = impacts_all[impacts_all.column.str.startswith(self.lprefix)]

        impact_df = impacts_all[impacts_all.id.isin(ids)][['word_prefix', 'impact', 'id']]
        start_el = self.dataset[self.dataset.id.isin(ids)]
        res += self.evaluate_impacts(impact_df, add_before_perturbation)

        if variable_side == 'all':
            impacts_all = self.impacts_df[(self.impacts_df.conf == conf_name)]
            impacts_all = impacts_all[impacts_all.column.str.startswith(self.rprefix)]
            impact_df = impacts_all[impacts_all.id.isin(ids)][['word_prefix', 'impact', 'id']]
            start_el = self.dataset[self.dataset.id.isin(ids)]
            res += self.evaluate_impacts(impact_df, add_before_perturbation)

        res_df = pd.DataFrame(res)
        res_df['conf'] = conf_name
        res_df['error'] = res_df.expected_delta - res_df.detected_delta
        return res_df

    def generate_evaluation(self, ids, fixed: str, overlap=True, **argv):
        evaluation_res = {}
        if fixed == 'right':
            fixed, f = 'right', 'R'
            variable, v = 'left', 'L'
        elif fixed == 'left':
            fixed, f = 'left', 'L'
            variable, v = 'right', 'R'
        else:
            assert False
        ov = '' if overlap == True else 'NOV'

        conf_name = f'{f}_{v}+{f}before{ov}'
        res_df = self.evaluate_set(ids, conf_name, fixed_side=fixed, variable_side=variable,
                                   add_before_perturbation=fixed, overlap=overlap, **argv)
        evaluation_res[conf_name] = res_df

        """
        conf_name = f'{f}_{f}+{v}after{ov}'
        res_df = self.evaluate_set(ids, conf_name, fixed_side=fixed, variable_side=fixed,
                                   add_after_perturbation=variable,
                                   overlap=overlap, **argv)
        evaluation_res[conf_name] = res_df
        """

        return evaluation_res

    def evaluation_routine(self, ids, **argv):
        assert np.all([x in self.impacts_df.id.unique() and x in self.dataset.id.unique() for x in ids]), \
            f'Missing some explanations {[x for x in ids if x in self.impacts_df.id.unique() or x in self.dataset.id.unique()]}'
        evaluations_dict = self.generate_evaluation(ids, fixed='right', overlap=True, **argv)
        evaluations_dict.update(self.generate_evaluation(ids, fixed='right', overlap=False, **argv))
        evaluations_dict.update(self.generate_evaluation(ids, fixed='left', overlap=True, **argv))
        evaluations_dict.update(self.generate_evaluation(ids, fixed='left', overlap=False, **argv))
        res_df = self.evaluate_set(ids, 'LIME', variable_side='all', fixed_side='', **argv)
        evaluations_dict['LIME'] = res_df
        res_df = self.evaluate_set(ids, 'left', variable_side='left', fixed_side='right', **argv)
        evaluations_dict['left'] = res_df
        res_df = self.evaluate_set(ids, 'right', variable_side='right', fixed_side='left', **argv)
        evaluations_dict['right'] = res_df
        res_df = self.evaluate_set(ids, 'mojito_copy_R', variable_side='right', fixed_side='left', **argv)
        evaluations_dict['mojito_copy_R'] = res_df
        res_df = self.evaluate_set(ids, 'mojito_copy_L', variable_side='left', fixed_side='right', **argv)
        evaluations_dict['mojito_copy_L'] = res_df

        return pd.concat(list(evaluations_dict.values()))


conf_code_map = {'all': 'all',
                 'R_L+Rafter': 'X_Y+Xafter', 'L_R+Lafter': 'X_Y+Xafter',
                 'R_R+Lafter': 'X_X+Yafter', 'L_L+Rafter': 'X_X+Yafter',
                 'R_L+Rbefore': 'X_Y+Xbefore', 'L_R+Lbefore': 'X_Y+Xbefore',
                 'R_L+RafterNOV': 'X_Y+XafterNOV', 'L_R+LafterNOV': 'X_Y+XafterNOV',
                 'R_L+RbeforeNOV': 'X_Y+XbeforeNOV', 'L_R+LbeforeNOV': 'X_Y+XbeforeNOV',
                 'R_R+LafterNOV': 'X_X+YafterNOV', 'L_L+RafterNOV': 'X_X+YafterNOV',
                 'left': 'X_Y', 'right': 'X_Y',
                 'leftCopy': 'X_YCopy', 'rightCopy': 'X_YCopy',
                 'mojito_copy_R': 'mojito_copy', 'mojito_copy_L': 'mojito_copy',
                 'mojito_drop': 'mojito_drop',
                 'LIME': 'LIME',
                 'MOJITO': 'MOJITO',
                 'LEMON': 'LEMON',
                 }


def evaluate_explanation_positive(impacts_match, explainer, num_round=25, utility=False):
    evaluation_res = {}
    ev = EvaluateExplanation(impacts_match, explainer.dataset, predict_method=explainer.model_predict,
                             exclude_attrs=explainer.exclude_attrs, percentage=.25, num_rounds=num_round)

    ids = impacts_match.query('conf =="LIME"').id.unique()

    conf_name = 'LIME'
    res_df = ev.evaluate_set(ids, conf_name, variable_side='all', utility=utility)
    evaluation_res[conf_name] = res_df

    conf_name = 'left'
    res_df = ev.evaluate_set(ids, conf_name, variable_side='left', fixed_side='right', utility=utility)
    evaluation_res[conf_name] = res_df

    conf_name = 'right'
    res_df = ev.evaluate_set(ids, conf_name, variable_side='right', fixed_side='left', utility=utility)
    evaluation_res[conf_name] = res_df

    conf_name = 'leftCopy'
    res_df = ev.evaluate_set(ids, conf_name, variable_side='left', fixed_side='right', add_before_perturbation='right',
                             overlap=False,
                             utility=utility)
    evaluation_res[conf_name] = res_df

    conf_name = 'rightCopy'
    res_df = ev.evaluate_set(ids, conf_name, variable_side='right', fixed_side='left', add_before_perturbation='left',
                             overlap=False,
                             utility=utility)
    evaluation_res[conf_name] = res_df

    tmp_df = pd.concat(list(evaluation_res.values()))
    tmp_df['conf_code'] = tmp_df.conf.map(conf_code_map)

    return aggregate_results(tmp_df, utility)


def evaluate_explanation_negative(impacts, explainer, num_round=25, utility=False):
    evaluation_res = {}

    ids = impacts.query('conf =="LIME"').id.unique()
    ev = EvaluateExplanation(impacts, explainer.dataset, predict_method=explainer.model_predict,
                             exclude_attrs=explainer.exclude_attrs, percentage=.25, num_rounds=num_round)

    conf_name = 'LIME'
    res_df = ev.evaluate_set(ids, conf_name, variable_side='all', utility=utility)
    evaluation_res[conf_name] = res_df

    conf_name = 'left'
    res_df = ev.evaluate_set(ids, conf_name, variable_side='left', fixed_side='right', utility=utility)
    evaluation_res[conf_name] = res_df

    conf_name = 'right'
    res_df = ev.evaluate_set(ids, conf_name, variable_side='right', fixed_side='left', utility=utility)
    evaluation_res[conf_name] = res_df
    conf_name = 'mojito_copy_L'
    res_df = ev.evaluate_set(ids, conf_name, variable_side='left', fixed_side='right', add_before_perturbation='right',
                             overlap=False,
                             utility=utility)
    evaluation_res[conf_name] = res_df

    conf_name = 'mojito_copy_R'
    res_df = ev.evaluate_set(ids, conf_name, variable_side='right', fixed_side='left', add_before_perturbation='left',
                             overlap=False,
                             utility=utility)
    evaluation_res[conf_name] = res_df

    conf_name = 'leftCopy'
    res_df = ev.evaluate_set(ids, conf_name, variable_side='left', fixed_side='right', add_before_perturbation='right',
                             overlap=False,
                             utility=utility)
    evaluation_res[conf_name] = res_df

    conf_name = 'rightCopy'
    res_df = ev.evaluate_set(ids, conf_name, variable_side='right', fixed_side='left', add_before_perturbation='left',
                             overlap=False,
                             utility=utility)
    evaluation_res[conf_name] = res_df

    tmp_df = pd.concat(list(evaluation_res.values()))
    tmp_df['conf_code'] = tmp_df.conf.map(conf_code_map)
    return aggregate_results(tmp_df, utility)


def aggregate_results(tmp_df, utility=False):
    if utility is False or utility == 'all':
        tmp_res = tmp_df
        tmp = tmp_res.groupby(['comb_name', 'conf_code']).apply(lambda x: pd.Series(
            {'accuracy': x[x.correct == True].shape[0] / x.shape[0], 'mae': x.error.abs().mean()})).reset_index()
        tmp.melt(['conf_code', 'comb_name']).set_index(['comb_name', 'conf_code', 'variable']).unstack(
            'conf_code').plot(kind='bar', figsize=(16, 6), rot=45);
    else:
        tmp_res = tmp_df
        tmp_res = tmp_res[
            tmp_res.comb_name.isin(['change_class', 'all_opposite']) | tmp_res.comb_name.str.startswith('change_class')]
        tmp_res['utility_base'] = (tmp_res['start_pred'] > .5) != (
                tmp_res['start_pred'] - tmp_res['expected_delta'] > .5)
        tmp_res['utility_model'] = (tmp_res['start_pred'] > .5) != (tmp_res['new_pred'] > .5)
        tmp_res['utility_and'] = tmp_res['utility_model'] & tmp_res['utility_base']
        tmp_res['U_baseFalse_modelTrue'] = (tmp_res['utility_base'] == False) & (tmp_res['utility_model'] == True)
        tmp = tmp_res.groupby(['id', 'comb_name', 'conf_code']).apply(lambda x: pd.Series(
            {'accuracy': x.correct.mean(), 'utility_and': x.utility_and.mean(),
             'mae': x.error.abs().mean()})).reset_index()
        tmp = tmp.groupby(['comb_name', 'conf_code'])['accuracy', 'mae', 'utility_and'].agg(
            ['mean', 'std']).reset_index()
        tmp.columns = [f"{a}{'_' + b if b else ''}" for a, b in tmp.columns]

    return tmp, tmp_res
