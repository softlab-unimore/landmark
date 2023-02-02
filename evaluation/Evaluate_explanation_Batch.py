import gc
import os
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
from tqdm.autonotebook import tqdm
from Landmark_github.evaluation.dataset_names import conf_code_map
from ..landmark.landmark import Landmark, Mapper

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


def append_prefix(word_relevance, df, decision_unit_view=False, exclude_attrs=['id', 'left_id', 'right_id', 'label']):
    ids = word_relevance['id'].unique()
    res_df = []
    for id in ids:
        el = df[df.id == id]
        word_relevance_el = word_relevance[word_relevance.id == id]
        if decision_unit_view is True:
            word_relevance_el['left_word_prefixes'] = get_prefix(word_relevance_el, el, 'left')
            word_relevance_el['right_word_prefixes'] = get_prefix(word_relevance_el, el, 'right')
        res_df.append(word_relevance_el.copy())
    res_df = pd.concat(res_df)
    if decision_unit_view is True:
        mapper = Mapper(df.loc[:, np.setdiff1d(df.columns, exclude_attrs)], r' ')
        assert len(mapper.attr_map.keys()) % 2 == 0, 'The attributes must be the same for the two sources.'
        shift = int(len(mapper.attr_map.keys()) / 2)
        res_df['right_word_prefixes'] = res_df['right_word_prefixes'].apply(
            lambda x: chr(ord(x[0]) + shift) + x[1:] if x != '[UNP]' else x)
    return res_df


def evaluate_df(word_relevance, df_to_process, predictor, exclude_attrs=['id', 'left_id', 'right_id', 'label'],
                score_col='pred', decision_unit_view=True, k=5):
    print(f'Testing unit remotion with -- {score_col}')
    assert df_to_process.shape[
               0] > 0, f'DataFrame to evaluate must have some elements. Passed df has shape {df_to_process.shape[0]}'
    evaluation_df = df_to_process.copy().replace(pd.NA, '')
    word_relevance_prefix = append_prefix(word_relevance, evaluation_df, exclude_attrs=exclude_attrs,
                                          decision_unit_view=decision_unit_view,)
    if score_col == 'pred':
        word_relevance_prefix['impact'] = word_relevance_prefix[score_col] - 0.5
    else:
        word_relevance_prefix['impact'] = word_relevance_prefix[score_col]
    word_relevance_prefix['conf'] = 'bert'

    res_list = []
    # for side in ['left', 'right']:
    # evaluation_df['pred'] = predictor(evaluation_df)
    side_word_relevance_prefix = word_relevance_prefix.copy()
    # side_word_relevance_prefix['word_prefix'] = side_word_relevance_prefix[side + '_word_prefixes']
    # side_word_relevance_prefix = side_word_relevance_prefix.query(f'{side}_word != "[UNP]"')
    ev = Evaluate_explanation(side_word_relevance_prefix, evaluation_df, predict_method=predictor,
                              exclude_attrs=exclude_attrs, percentage=.25, num_round=3)

    # fixed_side = 'right' if side == 'left' else 'left'
    res_df = ev.evaluate_set(df_to_process.id.values, 'bert', variable_side='all', fixed_side='all',
                             utility=True, k=k)
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
    explainer = Landmark(predictor, df, exclude_attrs=['id', 'label'], lprefix='left_', rprefix='right_')
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

from landmark.landmark import Landmark


class Evaluate_explanation(Landmark):

    def __init__(self, impacts_df, dataset, percentage=.25, num_round=10, decision_unit_view=False,
                 remove_decision_unit_only=False, **argv):
        self.impacts_df = impacts_df
        self.percentage = percentage
        self.num_round = num_round
        self.decision_unit_view = decision_unit_view
        self.remove_decision_unit_only = remove_decision_unit_only
        super().__init__(dataset=dataset, **argv)

    def prepare_impacts(self, impacts_df, start_el, variable_side, fixed_side,
                        add_before_perturbation, add_after_perturbation, overlap):
        self.words_with_prefixes = []
        self.impacts = []
        self.variable_encoded = []
        self.fixed_data_list = []
        for id in start_el.id.unique():
            impacts_sorted = impacts_df.query(f'id == {id}').sort_values('impact', ascending=False).reset_index(
                drop=True)
            self.impacts.append(impacts_sorted['impact'].values)

            if self.remove_decision_unit_only is True:
                self.words_with_prefixes.append(impacts_sorted)
                turn_vairable_encoded = impacts_sorted
                self.fixed_data = None
            else:
                if self.decision_unit_view is True:
                    self.words_with_prefixes.append(
                        [impacts_sorted['left_word_prefixes'].values, impacts_sorted['right_word_prefixes'].values])
                else:
                    self.words_with_prefixes.append(impacts_sorted['word_prefix'].values)

                turn_vairable_encoded = self.prepare_element(start_el[start_el.id == id].copy(), variable_side,
                                                             fixed_side,
                                                             add_before_perturbation, add_after_perturbation, overlap)
            self.fixed_data_list.append(self.fixed_data)
            self.variable_encoded.append(turn_vairable_encoded)

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
        df_list = []
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

        self.prepare_impacts(impacts_df, start_el, variable_side, fixed_side, add_before_perturbation,
                             add_after_perturbation, overlap)


        data_list = []
        description_to_evaluate_list = []
        for index, id_ in enumerate(start_el.id.unique()):
            all_comb = {}
            if utility is False:
                turn_comb = self.get_tokens_to_remove(self.start_pred[index], self.words_with_prefixes[index],
                                                      self.impacts[index])
                all_comb.update(**turn_comb)
            if utility is True or utility == 'all':
                change_class_tokens = self.get_tokens_to_change_class(self.start_pred[index], self.impacts[index])
                turn_comb = {'change_class': [change_class_tokens],
                             'single_word': [[x] for x in np.arange(self.impacts[index].shape[0])],
                             'all_opposite': [[pos for pos, impact in enumerate(self.impacts[index]) if
                                               (impact > 0) == (self.start_pred[index] > .5)]]}
                turn_comb['change_class_D.10'] = [
                    self.get_tokens_to_change_class(self.start_pred[index], self.impacts[index], delta=.1)]
                turn_comb['change_class_D.15'] = [
                    self.get_tokens_to_change_class(self.start_pred[index], self.impacts[index], delta=.15)]
                all_comb.update(**turn_comb)
            if utility == 'AOPC' or utility == 'all':
                turn_comb = self.get_tokens_to_remove_AOPC(self.start_pred[index], self.words_with_prefixes[index],
                                                           self.impacts[index], k=k)
                all_comb.update(**turn_comb)
            if utility == 'sufficiency' or utility == 'all':
                turn_comb = self.get_tokens_to_remove_sufficiency(self.start_pred[index],
                                                                  self.words_with_prefixes[index], self.impacts[index],
                                                                  k=k)
                all_comb.update(**turn_comb)
            if utility == 'degradation' or utility == 'all':
                turn_comb = self.get_tokens_to_remove_degradation(self.start_pred[index],
                                                                  self.words_with_prefixes[index], self.impacts[index],
                                                                  k=k)
                all_comb.update(**turn_comb)
            if utility == 'single_units' or utility == 'all':
                turn_comb = self.get_tokens_to_remove_single_units(self.start_pred[index],
                                                                   self.words_with_prefixes[index], self.impacts[index],
                                                                   k=k)
                all_comb.update(**turn_comb)
            res = self.generate_descriptions(all_comb, self.words_with_prefixes[index], self.variable_encoded[index])
            description_to_evaluate, comb_name_sequence, tokens_to_remove_sequence = res
            data_list.append([description_to_evaluate, comb_name_sequence, tokens_to_remove_sequence])
            self.data_list = data_list
            description_to_evaluate_list.append(description_to_evaluate)

        if self.fixed_data_list[0] is not None:
            self.batch_fixed_data = pd.concat(
                [self.fixed_data_list[i] for i, x in enumerate(description_to_evaluate_list) for l in range(len(x))])
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
        for index, id in enumerate(start_el.id.unique()):
            evaluation = {'id': id, 'start_pred': self.start_pred[index]}
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

    def get_tokens_to_remove(self, start_pred, tokens_sorted, impacts_sorted):
        if len(impacts_sorted) >= 5:
            combination = {'firts1': [[0]], 'first2': [[0, 1]], 'first5': [[0, 1, 2, 3, 4]]}
        else:
            combination = {'firts1': [[0]]}

        tokens_to_remove = self.get_tokens_to_change_class(start_pred, impacts_sorted)
        combination['change_class'] = [tokens_to_remove]
        lent = len(impacts_sorted)
        ntokens = int(lent * self.percentage)
        np.random.seed(0)
        combination['random'] = [np.random.choice(lent, ntokens, ) for _ in range(self.num_round)]
        return combination

    def get_tokens_to_change_class(self, start_pred, impacts_sorted, delta: float = 0.0):
        tokens_to_remove = []
        positive_match = start_pred > .5
        # delta = -delta if not positive else delta
        index = np.arange(0, len(impacts_sorted))

        if not positive_match:
            index = index[::-1]  # start removing negative impacts to push the score towards match if not positive

        delta_score_to_achieve = abs(start_pred - 0.5) + delta
        current_delta_score = 0

        for i in index:
            current_token_impact = impacts_sorted[i] * (1 if positive_match else -1)
            # remove positive impact if element is match, neg impacts if no match
            if current_token_impact > 0:
                tokens_to_remove.append(i)
                current_delta_score += current_token_impact
            else:  # there are no more tokens with positive (negative) impacts
                break

            # expected_delta = np.abs(np.sum(impacts_sorted[tokens_to_remove]))

            if current_delta_score >= delta_score_to_achieve:
                break

        return tokens_to_remove

    def get_tokens_to_remove_AOPC(self, start_pred, tokens_sorted, impacts_sorted, k=10):
        min_tokens = min(len(impacts_sorted), k)
        combination = {f'MoRF_{i}': [np.arange(i)] for i in range(1, min_tokens + 1)}
        np.random.seed(0)
        lent = len(impacts_sorted)
        for turn_n_tokens in range(1, min_tokens + 1):
            combination[f'random_{turn_n_tokens}'] = [np.random.choice(lent, turn_n_tokens, replace=False) for _ in
                                                      range(self.num_round)]
        return combination

    def get_tokens_to_remove_degradation(self, start_pred, tokens_sorted, impacts_sorted, k=100, random=False):
        lent = len(impacts_sorted)
        min_tokens = lent
        if start_pred > .5:
            combination = {f'MoRF_{i}': [np.arange(i)] for i in range(1, min_tokens + 1)}
            combination.update(**{f'LeRF_{i}': [np.arange(lent - i, lent)] for i in range(1, min_tokens + 1)})
        else:
            combination = {f'MoRF_{i}': [np.arange(lent - i, lent)] for i in range(1, min_tokens + 1)}
            combination.update(**{f'LeRF_{i}': [np.arange(i)] for i in range(1, min_tokens + 1)})
        np.random.seed(0)
        if random is True:
            for turn_n_tokens in range(1, min_tokens + 1):
                combination[f'random_{turn_n_tokens}'] = [np.random.choice(lent, turn_n_tokens, replace=False) for _ in
                                                          range(self.num_round)]
        return combination

    def get_tokens_to_remove_sufficiency(self, start_pred, tokens_sorted, impacts_sorted, k=10):
        lent = len(impacts_sorted)
        min_tokens = min(lent, k)
        if start_pred > .5:
            combination = {f'top_{i}': [np.arange(i, lent)] for i in range(1, min_tokens + 1)}
        else:
            combination = {f'top_{i}': [np.arange(lent - i)] for i in range(1, min_tokens + 1)}
        np.random.seed(0)
        for turn_n_tokens in range(1, min_tokens + 1):
            combination[f'random_{turn_n_tokens}'] = [
                np.setdiff1d(np.arange(lent), np.random.choice(lent, turn_n_tokens, replace=False)) for _ in
                range(self.num_round)]
        return combination

    def get_tokens_to_remove_single_units(self, start_pred, tokens_sorted, impacts_sorted, k=10):
        lent = len(impacts_sorted)
        combination = {f'unit_{i}': [[i]] for i in range(lent)}
        return combination

    def evaluate_set(self, ids, conf_name, variable_side='all', fixed_side='all', add_before_perturbation=None,
                     add_after_perturbation=None, overlap=True, utility=False,k=5,):
        """
        Batch of evaluate_df
        """
        res = []
        # if variable_side == 'all':
        #     impacts_all = impacts_all[impacts_all.column.str.startswith(self.lprefix)]

        # impact_df = impacts_all[impacts_all.id.isin(ids)][['word_prefix', 'impact', 'id']]
        # start_el = self.dataset[self.dataset.id.isin(ids)]
        # res += self.evaluate_impacts(start_el, impact_df, variable_side, fixed_side, add_before_perturbation,
        #                              add_after_perturbation, overlap, utility,k=k)

        # if variable_side == 'all':
        impacts_all = self.impacts_df[(self.impacts_df.conf == conf_name)]
        # impacts_all = impacts_all[impacts_all.column.str.startswith(self.rprefix)]
        impact_df = impacts_all[impacts_all.id.isin(ids)] #[['word_prefix', 'impact', 'id']]
        start_el = self.dataset[self.dataset.id.isin(ids)]
        res += self.evaluate_impacts(start_el, impact_df, variable_side, fixed_side,
                                     add_before_perturbation,
                                     add_after_perturbation, overlap, utility,k=k)

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


def evaluate_explanation_positive(impacts_match, explainer, num_round=25, utility=False):
    evaluation_res = {}
    ev = Evaluate_explanation(impacts_match, explainer.dataset, predict_method=explainer.model_predict,
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
    ev = Evaluate_explanation(impacts, explainer.dataset, predict_method=explainer.model_predict,
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

def calculate_save_metric(res_df, model_files_path, metric_name, prefix='', suffix=''):
    prefix = prefix + '_' if prefix != '' else ''
    suffix = ('_' + suffix) if suffix != '' else ''

    res_df['concorde'] = (res_df['detected_delta'] > 0) == (res_df['expected_delta'] > 0)
    res_df['same_class'] = (res_df['start_pred'] > 0.5) == (res_df['new_pred'] > .5)
    res_df['comb'] = res_df['comb_name']
    res_df['n_units_changed'] = 1
    mask = res_df['comb_name'].str.contains('|'.join(['MoRF', 'LeRF', 'random']))
    res_df.loc[mask, 'comb'] = res_df.loc[mask, 'comb_name'].apply(lambda x: x.split('_')[0])
    res_df.loc[mask, 'n_units_changed'] = res_df.loc[mask, 'comb_name'].apply(lambda x: x.split('_')[1])
    if metric_name == 'AOPC' or metric_name == 'all':
        grouped = res_df.groupby(['id', 'comb']).agg(
            {'detected_delta': [('sum', lambda x: x.sum()), 'size']}).droplevel(0, 1)
        AOPC = (grouped['sum'] / grouped['size']).groupby('comb').mean()
        tmp_path = os.path.join(model_files_path, f'{prefix}_AOPC_score{suffix}.csv')
        AOPC.to_csv(tmp_path)
        print(AOPC)
    if metric_name == 'sufficiency' or metric_name == 'all':
        sufficiency = res_df.groupby(['id', 'comb']).agg(
            {'same_class': [('sum', lambda x: x.sum() / x.size)]}).groupby('comb').mean()
        # assert False
        tmp_path = os.path.join(model_files_path, f'{prefix}_sufficiency_score{suffix}.csv')
        sufficiency.to_csv(tmp_path)
        print(sufficiency)

def aggregate_results(res_df, model_files_path, utility=False, score_col='pred', prefix=''):
    if utility is True or utility == 'all':
        res_df['concorde'] = (res_df['detected_delta'] > 0) == (res_df['expected_delta'] > 0)
        match_stat = res_df.groupby('comb_name')[['concorde']].mean()
        match_stat.to_csv(os.path.join(model_files_path, 'results', f'{prefix}_{score_col}__evaluation.csv'))
        match_stat = res_df.groupby('comb_name')[['detected_delta']].agg(['size', 'mean', 'median', 'min', 'max'])
        match_stat.to_csv(os.path.join(model_files_path, 'results', f'{prefix}_{score_col}__evaluation_mean_delta.csv'))

    if utility is False or utility == 'all':
        tmp_res = res_df
        tmp = tmp_res.groupby(['comb_name', 'conf_code']).apply(lambda x: pd.Series(
            {'accuracy': x[x.correct == True].shape[0] / x.shape[0], 'mae': x.error.abs().mean()})).reset_index()
        tmp.melt(['conf_code', 'comb_name']).set_index(['comb_name', 'conf_code', 'variable']).unstack(
            'conf_code').plot(kind='bar', figsize=(16, 6), rot=45);
    else:
        tmp_res = res_df
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
