import gc
import re
from functools import partial

import numpy as np
import pandas as pd
import plotly.express as px
import torch
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from tqdm.notebook import tqdm

from .removal_strategies import get_tokens_to_remove_five, get_tokens_to_remove_incrementally, \
    get_tokens_to_remove_aopc, get_tokens_to_remove_degradation, get_tokens_to_remove_sufficiency, \
    get_tokens_to_remove_single_units, get_tokens_to_change_class

def get_prefix(word_relevance_df, el, side: str):
    assert side in ['left', 'right']
    word_relevance_el = word_relevance_df.copy().reset_index(drop=True)
    mapper = Mapper([x for x in el.columns if x.startswith(side + '_') and x != side + '_id'], r' ')
    available_prefixes = mapper.encode_attr(el).split()
    assigned_pref = []
    word_prefixes = []
    attr_to_code = {v: k for k, v in mapper.attr_map.items()}
    for i in range(word_relevance_el.shape[0]):
        word = str(word_relevance_el.loc[i, side + '_word'])
        if word == '[UNP]':
            word_prefixes.append('[UNP]')
        else:
            col = word_relevance_el.loc[i, side + '_attribute']
            col_code = attr_to_code[side + '_' + col]
            turn_prefixes = [x for x in available_prefixes if x[0] == col_code]
            idx = 0
            while idx < len(turn_prefixes) and word != turn_prefixes[idx][4:]:
                idx += 1
            if idx < len(turn_prefixes):
                tmp = turn_prefixes[idx]
                del turn_prefixes[idx]
                word_prefixes.append(tmp)
                assigned_pref.append(tmp)
            else:
                idx = 0
                while idx < len(assigned_pref) and word != assigned_pref[idx][4:]:
                    idx += 1
                if idx < len(assigned_pref):
                    word_prefixes.append(assigned_pref[idx])
                else:
                    assert False, word

    return word_prefixes


def append_prefix(em_df, word_relevance, decision_unit_view=False,
                  exclude_attrs=('id', 'left_id', 'right_id', 'label')):
    ids = word_relevance['id'].unique()
    res_df = list()
    for id in ids:
        el = em_df[em_df.id == id]
        word_relevance_el = word_relevance[word_relevance.id == id]
        if decision_unit_view is True:
            word_relevance_el['left_word_prefixes'] = get_prefix(word_relevance_el, el, 'left')
            word_relevance_el['right_word_prefixes'] = get_prefix(word_relevance_el, el, 'right')
        res_df.append(word_relevance_el.copy())
    res_df = pd.concat(res_df)
    if decision_unit_view is True:
        mapper = Mapper(em_df.loc[:, np.setdiff1d(em_df.columns, exclude_attrs)], r' ')
        assert len(mapper.attr_map.keys()) % 2 == 0, 'The attributes must be the same for the two sources.'
        shift = int(len(mapper.attr_map.keys()) / 2)
        res_df['right_word_prefixes'] = res_df['right_word_prefixes'].apply(
            lambda x: chr(ord(x[0]) + shift) + x[1:] if x != '[UNP]' else x)

    return res_df


def evaluate_df(word_relevance, df_to_process, predictor, exclude_attrs=('id', 'left_id', 'right_id', 'label'),
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

    res_list = []
    for side in ['left', 'right']:
        evaluation_df['pred'] = predictor(evaluation_df)
        side_word_relevance_prefix = word_relevance_prefix.copy()
        side_word_relevance_prefix['word_prefix'] = side_word_relevance_prefix[side + '_word_prefixes']
        side_word_relevance_prefix = side_word_relevance_prefix.query(f'{side}_word != "[UNP]"')
        ev = EvaluateExplanation(side_word_relevance_prefix, evaluation_df, predict_method=predictor,
                                 exclude_attrs=exclude_attrs, percentage=.25, num_round=3)

        fixed_side = 'right' if side == 'left' else 'left'
        res_df = ev.evaluate_set(df_to_process.id.values, 'bert', variable_side=side, fixed_side=fixed_side,
                                 utility=True)
        res_list.append(res_df.copy())

    return pd.concat(res_list)


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


def token_remotion_delta_performance(df, y_true, word_relevance, predictor, k_list=[10, 5, 3, 1], plot=True,
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


class Landmark(object):

    def __init__(self, predict_method, dataset, exclude_attrs=('id', 'label'), split_expression=' ',
                 lprefix='left_', rprefix='right_', **argv):
        """
        :param predict_method: of the model to be explained
        :param dataset: containing the elements that will be explained. Used to save the attribute structure.
        :param exclude_attrs: attributes to be excluded from the explanations
        :param split_expression: to divide tokens from string
        :param lprefix: left prefix
        :param rprefix: right prefix
        :param argv: other optional parameters that will be passed to LIME
        """
        self.splitter = re.compile(split_expression)
        self.split_expression = split_expression
        self.explainer = LimeTextExplainer(class_names=['NO match', 'MATCH'], split_expression=split_expression, **argv)
        self.model_predict = predict_method
        self.dataset = dataset
        self.lprefix = lprefix
        self.rprefix = rprefix
        self.exclude_attrs = exclude_attrs

        self.cols = [x for x in dataset.columns if x not in exclude_attrs]
        self.left_cols = [x for x in self.cols if x.startswith(self.lprefix)]
        self.right_cols = [x for x in self.cols if x.startswith(self.rprefix)]
        self.cols = self.left_cols + self.right_cols
        self.explanations = {}

        self.add_after_perturbation = None
        self.overlap = None
        self.mapper_variable = None
        self.fixed_data = None
        self.fixed_side = str()

    def explain(self, elements, conf='auto', num_samples=500, verbose=False, **argv):
        """
        User interface to generate an explanations with the specified configurations for the elements passed in input.
        """
        assert type(elements) == pd.DataFrame, f'elements must be of type {pd.DataFrame}'
        allowed_conf = ['auto', 'single', 'double', 'LIME']
        assert conf in allowed_conf, 'conf must be in ' + repr(allowed_conf)
        if elements.shape[0] == 0:
            return None

        if 'auto' == conf:
            match_elements = elements[elements.label == 1]
            no_match_elements = elements[elements.label == 0]
            match_explanation = self.explain(match_elements, 'single', num_samples, **argv)
            no_match_explanation = self.explain(no_match_elements, 'double', num_samples, **argv)
            return pd.concat([match_explanation, no_match_explanation])

        to_cycle = tqdm(range(elements.shape[0])) if verbose else range(elements.shape[0])
        impact_list = []
        if 'LIME' == conf:
            for idx in to_cycle:
                impacts = self.explain_instance(elements.iloc[[idx]], variable_side='all', fixed_side='',
                                                num_samples=num_samples, **argv)
                impacts['conf'] = 'LIME'
                impact_list.append(impacts)
            self.impacts = pd.concat(impact_list)
            return self.impacts

        landmark = 'right'
        variable = 'left'
        overlap = False
        if 'single' == conf:
            add_before = None
        elif 'double' == conf:
            add_before = landmark
        else:
            raise ValueError

        # right landmark
        for idx in to_cycle:
            impacts = self.explain_instance(elements.iloc[[idx]], variable_side=variable, fixed_side=landmark,
                                            add_before_perturbation=add_before, num_samples=num_samples,
                                            overlap=overlap, **argv)
            impacts['conf'] = f'{landmark}_landmark' + ('_injection' if add_before is not None else '')
            impact_list.append(impacts)

        # switch sides
        landmark, variable = variable, landmark
        if add_before is not None:
            add_before = landmark

        # left landmark
        for idx in to_cycle:
            impacts = self.explain_instance(elements.iloc[[idx]], variable_side=variable, fixed_side=landmark,
                                            add_before_perturbation=add_before, num_samples=num_samples,
                                            overlap=overlap, **argv)
            impacts['conf'] = f'{landmark}_landmark' + ('_injection' if add_before is not None else '')
            impact_list.append(impacts)

        self.impacts = pd.concat(impact_list)
        return self.impacts

    def explain_instance(self, el, variable_side='left', fixed_side='right', add_before_perturbation=None,
                         add_after_perturbation=None, overlap=True, num_samples=500, **argv):
        """
        Main method to wrap the explainer and generate a landmark. A sort of Facade for the explainer.
        :param el: DataFrame containing the element to be explained.
        :return: landmark DataFrame
        """
        variable_el = el.copy()
        for col in self.cols:
            variable_el[col] = ' '.join(re.split(r' +', str(variable_el[col].values[0]).strip()))

        variable_data = self.prepare_element(variable_el, variable_side, fixed_side, add_before_perturbation,
                                             add_after_perturbation, overlap)

        words = self.splitter.split(variable_data)
        explanation = self.explainer.explain_instance(variable_data, self.restucture_and_predict,
                                                      num_features=len(words), num_samples=num_samples,
                                                      **argv)
        self.variable_data = variable_data  # to test the addition before perturbation

        id = el.id.values[0]  # Assume index is the id column
        self.explanations[f'{self.fixed_side}{id}'] = explanation
        return self.explanation_to_df(explanation, words, self.mapper_variable.attr_map, id)

    def prepare_element(self, variable_el, variable_side, fixed_side, add_before_perturbation, add_after_perturbation,
                        overlap):
        """
        Compute the data and set parameters needed to perform the landmark.
            Set fixed_side, fixed_data, mapper_variable.
            Call compute_tokens if needed
        """

        self.add_after_perturbation = add_after_perturbation
        self.overlap = overlap
        self.fixed_side = fixed_side
        if variable_side in ['left', 'right']:
            variable_cols = self.left_cols if variable_side == 'left' else self.right_cols

            assert fixed_side in ['left', 'right']

            if fixed_side == 'left':
                fixed_cols, not_fixed_cols = self.left_cols, self.right_cols
            else:
                fixed_cols, not_fixed_cols = self.right_cols, self.left_cols

            mapper_fixed = Mapper(fixed_cols, self.split_expression)
            self.fixed_data = mapper_fixed.decode_words_to_attr(mapper_fixed.encode_attr(
                variable_el[fixed_cols]))  # encode and decode data of fixed source to ensure the same format
            self.mapper_variable = Mapper(not_fixed_cols, self.split_expression)

            if add_before_perturbation is not None or add_after_perturbation is not None:
                self.compute_tokens(variable_el)
                if add_before_perturbation is not None:
                    self.add_tokens(variable_el, variable_cols, add_before_perturbation, overlap)
            variable_data = Mapper(variable_cols, self.split_expression).encode_attr(variable_el)

        elif variable_side == 'all':
            variable_cols = self.left_cols + self.right_cols

            self.mapper_variable = Mapper(variable_cols, self.split_expression)
            self.fixed_data = None
            self.fixed_side = 'all'
            variable_data = self.mapper_variable.encode_attr(variable_el)
        else:
            assert False, f'Not a feasible configuration. variable_side: {variable_side} not allowed.'

        return variable_data

    def explanation_to_df(self, explanation, words, attribute_map, id):
        """
        Generate the DataFrame of the landmark from the LIME landmark.
        :param explanation: LIME landmark
        :param words: words of the element subject of the landmark
        :param attribute_map: attribute map to decode the attribute from a prefix
        :param id: id of the element under landmark
        :return: DataFrame containing the landmark
        """
        impacts_list = []
        dict_impact = {'id': id}
        for wordpos, impact in explanation.as_map()[1]:
            word = words[wordpos]
            dict_impact.update(column=attribute_map[word[0]], position=int(word[1:3]), word=word[4:], word_prefix=word,
                               impact=impact)
            impacts_list.append(dict_impact.copy())
        return pd.DataFrame(impacts_list).reset_index()

    def compute_tokens(self, el):
        """
        Divide tokens of the descriptions for each column pair in inclusive and exclusive sets.
        :param el: pd.DataFrame containing the 2 description to analyze
        """
        tokens = {col: np.array(self.splitter.split(str(el[col].values[0]))) for col in self.cols}
        tokens_intersection = {}
        tokens_not_overlapped = {}
        for col in [col.replace('left_', '') for col in self.left_cols]:
            lcol, rcol = self.lprefix + col, self.rprefix + col
            tokens_intersection[col] = np.intersect1d(tokens[lcol], tokens[rcol])
            tokens_not_overlapped[lcol] = tokens[lcol][~ np.in1d(tokens[lcol], tokens_intersection[col])]
            tokens_not_overlapped[rcol] = tokens[rcol][~ np.in1d(tokens[rcol], tokens_intersection[col])]
        self.tokens_not_overlapped = tokens_not_overlapped
        self.tokens_intersection = tokens_intersection
        self.tokens = tokens
        return dict(tokens=tokens, tokens_intersection=tokens_intersection, tokens_not_overlapped=tokens_not_overlapped)

    def add_tokens(self, el, dst_columns, src_side, overlap=True):
        """
        Takes tokens computed before from the src_sside with overlap or not
        and inject them into el in columns specified in dst_columns.
        """
        if not overlap:
            tokens_to_add = self.tokens_not_overlapped
        else:
            tokens_to_add = self.tokens

        if src_side == 'left':
            src_columns = self.left_cols
        elif src_side == 'right':
            src_columns = self.right_cols
        else:
            assert False, f'src_side must "left" or "right". Got {src_side}'

        for col_dst, col_src in zip(dst_columns, src_columns):
            if len(tokens_to_add[col_src]) == 0:
                continue
            el[col_dst] = el[col_dst].astype(str) + ' ' + ' '.join(tokens_to_add[col_src])

    def restucture_and_predict(self, perturbed_strings):
        """
            Restructure the perturbed strings from LIME and return the related predictions.
        """
        non_null_rows = np.array([True] * len(perturbed_strings))
        if self.remove_decision_unit_only is True:
            for i, value in enumerate(perturbed_strings):
                if value.empty:
                    non_null_rows[i] = False
                else:
                    value['id'] = i
            self.tmp_dataset = pd.concat(perturbed_strings)

        else:
            self.tmp_dataset = self.restructure_strings(perturbed_strings)
            self.tmp_dataset.reset_index(inplace=True, drop=True)
        ret = np.ndarray(shape=(len(perturbed_strings), 2))
        ret[:, :] = 0.5
        predictions = self.model_predict(self.tmp_dataset)
        # assert len(perturbed_strings) == len(predictions), f'df and predictions shape are misaligned'

        ret[non_null_rows, 1] = np.array(predictions)
        ret[:, 0] = 1 - ret[:, 1]
        return ret

    def restructure_strings(self, perturbed_strings):
        """
        Decode :param perturbed_strings into DataFrame and
        :return reconstructed pairs appending the landmark entity.
        """
        df_list = []
        for single_row in perturbed_strings:
            df_list.append(self.mapper_variable.decode_words_to_attr_dict(single_row))
        variable_df = pd.DataFrame.from_dict(df_list)
        if self.add_after_perturbation is not None:
            self.add_tokens(variable_df, variable_df.columns, self.add_after_perturbation, overlap=self.overlap)
        if self.fixed_data is not None:
            fixed_df = pd.concat([self.fixed_data] * variable_df.shape[0])
            fixed_df.reset_index(inplace=True, drop=True)
        else:
            fixed_df = None
        return pd.concat([variable_df, fixed_df], axis=1)

    def double_explanation_conversion(self, explanation_df, item):
        """
        Compute and assign the original attribute of injected words.
        :return: explanation with original attribute for injected words.
        """
        view = explanation_df[['column', 'position', 'word', 'impact']].reset_index(drop=True)
        tokens_divided = self.compute_tokens(item)
        exchanged_idx = [False] * len(view)
        lengths = {col: len(words) for col, words in tokens_divided['tokens'].items()}
        for col, words in tokens_divided['tokens_not_overlapped'].items():  # words injected in the opposite side
            prefix, col_name = col.split('_')
            prefix = 'left_' if prefix == 'right' else 'right_'
            opposite_col = prefix + col_name
            exchanged_idx = exchanged_idx | ((view.position >= lengths[opposite_col]) & (view.column == opposite_col))
        exchanged = view[exchanged_idx]
        view = view[~exchanged_idx]
        # determine injected impacts
        exchanged['side'] = exchanged['column'].apply(lambda x: x.split('_')[0])
        col_names = exchanged['column'].apply(lambda x: x.split('_')[1])
        exchanged['column'] = np.where(exchanged['side'] == 'left', 'right_', 'left_') + col_names
        tmp = view.merge(exchanged, on=['word', 'column'], how='left', suffixes=('', '_injected'))
        tmp = tmp.drop_duplicates(['column', 'word', 'position'], keep='first')
        impacts_injected = tmp['impact_injected']
        impacts_injected = impacts_injected.fillna(0)

        view['score_right_landmark'] = np.where(view['column'].str.startswith('left'), view['impact'], impacts_injected)
        view['score_left_landmark'] = np.where(view['column'].str.startswith('right'), view['impact'], impacts_injected)
        view.drop('impact', 1, inplace=True)

        return view

    # def plot(self, explanation, el, figsize=(16, 6)):
    #     exp_double = self.double_explanation_conversion(explanation, el)
    #     PlotExplanation.plot(exp_double, figsize)


class Mapper(object):
    """
    This class is useful to encode a row of a dataframe in a string in which a prefix
    is added to each word to keep track of its attribute and its position.
    """

    def __init__(self, columns, split_expression):
        self.columns = columns
        self.attr_map = {chr(ord('A') + colidx): col for colidx, col in enumerate(self.columns)}
        self.arange = np.arange(100)
        self.split_expression = split_expression

    def decode_words_to_attr_dict(self, text_to_restructure):
        res = re.findall(r'(?P<attr>[A-Z]{1})(?P<pos>[0-9]{2})_(?P<word>[^' + self.split_expression + ']+)',
                         text_to_restructure)
        structured_row = {col: '' for col in self.columns}
        for col_code, pos, word in res:
            structured_row[self.attr_map[col_code]] += word + ' '
        for col in self.columns:  # Remove last space
            structured_row[col] = structured_row[col][:-1]
        return structured_row

    def decode_words_to_attr(self, text_to_restructure):
        return pd.DataFrame([self.decode_words_to_attr_dict(text_to_restructure)])

    def encode_attr(self, el):
        return ' '.join(
            [chr(ord('A') + colpos) + "{:02d}_".format(wordpos) + word for colpos, col in enumerate(self.columns) for
             wordpos, word in enumerate(re.split(self.split_expression, str(el[col].values[0])))])

    def encode_elements(self, elements):
        word_dict = {}
        res_list = []
        for i in np.arange(elements.shape[0]):
            el = elements.iloc[i]
            word_dict.update(id=el.id)
            for colpos, col in enumerate(self.columns):
                word_dict.update(column=col)
                for wordpos, word in enumerate(re.split(self.split_expression, str(el[col]))):
                    word_dict.update(word=word, position=wordpos,
                                     word_prefix=chr(ord('A') + colpos) + f"{wordpos:02d}_" + word)
                    res_list.append(word_dict.copy())
        return pd.DataFrame(res_list)


class EvaluateExplanation(Landmark):

    def __init__(self, dataset, impacts_df, percentage=.25, num_round=10, decision_unit_view=False,
                 remove_decision_unit_only=False, **argv):
        super().__init__(dataset=dataset, **argv)
        self.impacts_df = impacts_df
        self.percentage = percentage
        self.num_round = num_round
        self.decision_unit_view = decision_unit_view
        self.remove_decision_unit_only = remove_decision_unit_only
        self.impacts = list()
        self.words_with_prefixes = list()
        self.variable_encoded = list()
        self.fixed_data_list = list()
        self.fixed_data = None
        self.batch_fixed_data = None
        self.start_pred = None
        self.data_list = list()
        self.preds = None

    def prepare_impacts(self, start_el, impacts_df, variable_side, fixed_side, add_before_perturbation,
                        add_after_perturbation, overlap):
        for id_ in start_el.id.unique():
            turn_variable_encoded = None

            impacts_sorted = impacts_df.query(f'id == {id_}').sort_values('impact', ascending=False).reset_index(
                drop=True)
            self.impacts.append(impacts_sorted['impact'].values)

            if self.remove_decision_unit_only is True:
                if self.decision_unit_view:
                    self.words_with_prefixes.append(impacts_sorted)
                    turn_variable_encoded = impacts_sorted
                    self.fixed_data = None
            else:
                if self.decision_unit_view is True:
                    self.words_with_prefixes.append(
                        [impacts_sorted['left_word_prefixes'].values, impacts_sorted['right_word_prefixes'].values])
                else:
                    self.words_with_prefixes.append(impacts_sorted['word_prefix'].values)

                turn_variable_encoded = self.prepare_element(start_el[start_el.id == id_].copy(), variable_side,
                                                             fixed_side,
                                                             add_before_perturbation, add_after_perturbation, overlap)
            self.fixed_data_list.append(self.fixed_data)
            self.variable_encoded.append(turn_variable_encoded)

        if self.fixed_data_list[0] is not None:
            self.batch_fixed_data = pd.concat(self.fixed_data_list)
        else:
            self.batch_fixed_data = None
        # if variable_side == 'left' and add_before_perturbation is not None:
        #     assert False

        self.start_pred = self.restucture_and_predict(self.variable_encoded)[:, 1]  # match_score

    def restructure_strings(self, perturbed_strings):
        """

        Decode :param perturbed_strings into DataFrame and
        :return reconstructed pairs appending the landmark entity.

        """

        df_list = list()
        for single_row in perturbed_strings:
            df_list.append(self.mapper_variable.decode_words_to_attr_dict(single_row))
        variable_df = pd.DataFrame.from_dict(df_list)
        if self.add_after_perturbation is not None:
            self.add_tokens(variable_df, variable_df.columns, self.add_after_perturbation, overlap=self.overlap)
        if self.fixed_data is not None:
            fixed_df = self.batch_fixed_data
            fixed_df.reset_index(inplace=True, drop=True)
        else:
            fixed_df = None
        return pd.concat([variable_df, fixed_df], axis=1)

    def generate_descriptions(self, combinations_to_remove, words_with_prefixes, variable_encoded):
        description_to_evaluate = []
        comb_name_sequence = []
        tokens_to_remove_sequence = []
        for comb_name, combinations in combinations_to_remove.items():
            for tokens_to_remove in combinations:
                tmp_encoded = variable_encoded
                if self.decision_unit_view:  # remove both tokens of left and right as a united view without landmark
                    if self.remove_decision_unit_only:
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

    def evaluate_impacts(self, start_el, impacts_df, variable_side='left', fixed_side='right',
                         add_before_perturbation=None,
                         add_after_perturbation=None, overlap=True, utility=False, k=5):

        self.prepare_impacts(start_el, impacts_df, variable_side, fixed_side, add_before_perturbation,
                             add_after_perturbation, overlap)

        data_list = list()
        description_to_evaluate_list = list()
        for index, id_ in enumerate(start_el.id.unique()):
            all_comb = {}
            if utility is False:
                turn_comb = get_tokens_to_remove_five(self.start_pred[index], self.impacts[index],
                                                      self.percentage, self.num_round)
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
                turn_comb = get_tokens_to_remove_aopc(self.impacts[index], self.num_round, k=k)
                all_comb.update(**turn_comb)

            if utility == 'sufficiency' or utility == 'all':
                turn_comb = get_tokens_to_remove_sufficiency(self.start_pred[index], self.impacts[index],
                                                             self.num_round, k=k)
                all_comb.update(**turn_comb)

            if utility == 'degradation' or utility == 'all':
                turn_comb = get_tokens_to_remove_degradation(self.start_pred[index], self.impacts[index],
                                                             self.num_round)
                all_comb.update(**turn_comb)

            if utility == 'single_units' or utility == 'all':
                turn_comb = get_tokens_to_remove_single_units(self.impacts[index])
                all_comb.update(**turn_comb)

            if utility == 'incremental' or utility == 'all':
                turn_comb = get_tokens_to_remove_incrementally(self.impacts[index], limit=k)
                all_comb.update(**turn_comb)

            res = self.generate_descriptions(all_comb, self.words_with_prefixes[index], self.variable_encoded[index])
            description_to_evaluate, comb_name_sequence, tokens_to_remove_sequence = res
            data_list.append([description_to_evaluate, comb_name_sequence, tokens_to_remove_sequence])
            self.data_list = data_list
            description_to_evaluate_list.append(description_to_evaluate)

        if self.fixed_data_list[0] is not None:
            self.batch_fixed_data = pd.concat(
                [self.fixed_data_list[i] for i, x in enumerate(description_to_evaluate_list) for _ in range(len(x))])
        else:
            self.batch_fixed_data = None
        all_descriptions = np.concatenate(description_to_evaluate_list)
        preds = self.restucture_and_predict(all_descriptions)[:, 1]
        assert len(preds) == len(all_descriptions)
        splitted_preds = []
        start_idx = 0
        for turn_desc in description_to_evaluate_list:
            end_idx = start_idx + len(turn_desc)
            splitted_preds.append(preds[start_idx: end_idx])
            start_idx = end_idx
        self.preds = preds
        res_list = []
        for index, id_ in enumerate(start_el.id.unique()):
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

                if self.decision_unit_view is True:
                    if self.remove_decision_unit_only is True:
                        evaluation.update(tokens_removed=words_with_prefixes.loc[
                            tokens_to_remove, ['left_word_prefixes', 'right_word_prefixes']].values.tolist())
                    else:
                        evaluation.update(tokens_removed=list(
                            [list(turn_pref[tokens_to_remove]) for turn_pref in words_with_prefixes]))

                else:
                    evaluation.update(tokens_removed=list(words_with_prefixes[tokens_to_remove]))
                res_list.append(evaluation.copy())

        return res_list


    def evaluate_set(self, ids, conf_name, variable_side='left', fixed_side='right', add_before_perturbation=None,
                     add_after_perturbation=None, overlap=True, utility=False):
        impacts_all = self.impacts_df[(self.impacts_df.conf == conf_name)]
        res = []
        if variable_side == 'all':
            impacts_all = impacts_all[impacts_all.column.str.startswith(self.lprefix)]

        impact_df = impacts_all[impacts_all.id.isin(ids)][['word_prefix', 'impact', 'id']]
        start_el = self.dataset[self.dataset.id.isin(ids)]
        res += self.evaluate_impacts(start_el, impact_df, variable_side, fixed_side, add_before_perturbation,
                                     add_after_perturbation, overlap, utility)

        if variable_side == 'all':
            impacts_all = self.impacts_df[(self.impacts_df.conf == conf_name)]
            impacts_all = impacts_all[impacts_all.column.str.startswith(self.rprefix)]
            impact_df = impacts_all[impacts_all.id.isin(ids)][['word_prefix', 'impact', 'id']]
            start_el = self.dataset[self.dataset.id.isin(ids)]
            res += self.evaluate_impacts(start_el, impact_df, variable_side, fixed_side,
                                         add_before_perturbation,
                                         add_after_perturbation, overlap, utility)

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
        res_df = self.evaluate_set(ids, 'LIME', variable_side='all', fixed_side=None, **argv)
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
                             exclude_attrs=explainer.exclude_attrs, percentage=.25, num_round=num_round)

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
                             exclude_attrs=explainer.exclude_attrs, percentage=.25, num_round=num_round)

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
