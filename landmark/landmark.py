import re
from tqdm import tqdm
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
from .plot import PlotExplanation


class Landmark(object):

    def __init__(self, predict_method, dataset, exclude_attrs=('id', 'label'), split_expression=' ',
                 lprefix='left_', rprefix='right_', variable_side: str='all', fixed_side: str='all', 
                 add_before_perturbation=None, add_after_perturbation=None, overlap: bool=True, **kwargs):
        """

        :param predict_method: of the model to be explained
        :param dataset: containing the elements that will be explained. Used to save the attribute structure.
        :param exclude_attrs: attributes to be excluded from the explanations
        :param split_expression: to divide tokens from string
        :param lprefix: left prefix
        :param rprefix: right prefix
        :param kwargs: other optional parameters that will be passed to LIME
        """
        self.tokens = None
        self.tmp_dataset = None
        self.tokens_intersection = None
        self.tokens_not_overlapped = None
        self.variable_side = variable_side
        self.fixed_side = fixed_side
        self.variable_data = None
        self.fixed_data = None
        self.mapper = None
        self.overlap = overlap
        self.add_before_perturbation = add_before_perturbation
        self.add_after_perturbation = add_after_perturbation
        self.impacts = None

        self.splitter = re.compile(split_expression)
        self.split_expression = split_expression
        self.explainer = LimeTextExplainer(class_names=['NO match', 'MATCH'], split_expression=split_expression,
                                           **kwargs)
        self.model_predict = predict_method
        self.dataset = dataset
        self.lprefix = lprefix
        self.rprefix = rprefix
        self.exclude_attrs = exclude_attrs

        self.cols = [x for x in dataset.columns if x not in exclude_attrs]
        self.left_cols = [x for x in self.cols if x.startswith(self.lprefix)]
        self.right_cols = [x for x in self.cols if x.startswith(self.rprefix)]
        self.cols = self.left_cols + self.right_cols
        self.explanations = dict()

    def update_settings(self, **kwargs):
        updated_dataset = False
        updated_split_expression = False

        if 'predict_method' in kwargs:
            self.model_predict = kwargs['predict_method']

        if 'dataset' in kwargs:
            updated_dataset = True
            self.dataset = kwargs['dataset']

        if 'lprefix' in kwargs:
            self.lprefix = kwargs['lprefix']

        if 'rprefix' in kwargs:
            self.rprefix = kwargs['rprefix']

        if 'exclude_attrs' in kwargs:
            self.exclude_attrs = kwargs['exclude_attrs']

        if 'split_expression' in kwargs:
            updated_split_expression = True
            self.split_expression = kwargs['split_expression']

        if 'overlap' in kwargs:
            self.overlap = kwargs['overlap']
            
        # TODO: implement checks for variable and fixed sides
            
        if 'add_before_perturbation' in kwargs:
            self.add_before_perturbation = kwargs['add_before_perturbation']

        if 'add_after_perturbation' in kwargs:
            self.add_after_perturbation = kwargs['add_after_perturbation']
            
        if 'variable_side' in kwargs:
            self.variable_side = kwargs['variable_side']
            
        if 'fixed_side' in kwargs:
            self.fixed_side = kwargs['fixed_side']

        if updated_dataset:
            self.tokens = None
            self.tmp_dataset = None
            self.tokens_intersection = None
            self.tokens_not_overlapped = None
            self.variable_data = None
            self.fixed_data = None
            self.mapper = None
            self.impacts = None

            self.cols = [x for x in self.dataset.columns if x not in self.exclude_attrs]
            self.left_cols = [x for x in self.cols if x.startswith(self.lprefix)]
            self.right_cols = [x for x in self.cols if x.startswith(self.rprefix)]
            self.cols = self.left_cols + self.right_cols
            self.explanations = dict()

        if updated_split_expression:
            self.splitter = re.compile(self.split_expression)
            # TODO: What to check to reset the LimeTextExplainer?
            self.explainer = LimeTextExplainer(class_names=['NO match', 'MATCH'], split_expression=self.split_expression,
                                               **kwargs)

        print("Landmark settings updated.")

    def explain(self, elements, conf='auto', num_samples=500, **argv):
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

        impact_list = list()
        if 'LIME' == conf:
            for idx in tqdm(range(elements.shape[0])):
                impacts = self.explain_instance(elements.iloc[[idx]], variable_side='all', fixed_side='',
                                                num_samples=num_samples, **argv)
                impacts['conf'] = 'LIME'
                impact_list.append(impacts)
            self.impacts = pd.concat(impact_list)
            return self.impacts

        landmark = 'right'
        variable = 'left'
        overlap = False
        add_before = None  # default if conf == 'single'
        
        if conf == 'double':
            add_before = landmark

        # right landmark
        for idx in tqdm(range(elements.shape[0])):
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
        for idx in tqdm(range(elements.shape[0])):
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

        Args:
            el: DataFrame containing the element to be explained.
            num_samples: the number of samples to explain from the el DataFrame.
            overlap: boolean to overlap tokens or not in the explanation.
            add_after_perturbation: TO_COMPLETE
            add_before_perturbation: TO_COMPLETE
            fixed_side: the side to keep as it is, without perturbations. Can be 'left', 'right', or 'all'.
            variable_side: the side on which to apply perturbations. Can be 'left', 'right', or 'all'.
        Returns:
            landmark DataFrame
        """
        # TODO: complete docstring

        variable_el = el.copy()
        for col in self.cols:
            variable_el[col] = ' '.join(re.split(r' +', str(variable_el[col].values[0]).strip()))

        variable_data = self.prepare_element(variable_el)

        words = self.splitter.split(variable_data)
        explanation = self.explainer.explain_instance(variable_data, self.restructure_and_predict,
                                                      num_features=len(words), num_samples=num_samples,
                                                      **argv)
        self.variable_data = variable_data  # to test the addition before perturbation

        id_ = el.id.values[0]  # Assume index is the id column
        self.explanations[f'{self.fixed_side}{id_}'] = explanation
        return self.explanation_to_df(explanation, words, self.mapper.attr_map, id_)

    def prepare_element(self, variable_el):
        """
        Compute the data and set parameters needed to perform the landmark.
            Set fixed_side, fixed_data, mapper_variable.
            Call compute_tokens if needed
        """

        if self.variable_side in ['left', 'right']:
            variable_cols = self.left_cols if self.variable_side == 'left' else self.right_cols

            assert self.fixed_side in ['left', 'right']
            if self.fixed_side == 'left':
                fixed_cols, not_fixed_cols = self.left_cols, self.right_cols
            else:
                fixed_cols, not_fixed_cols = self.right_cols, self.left_cols
            mapper_fixed = Mapper(fixed_cols, self.split_expression)
            self.fixed_data = mapper_fixed.decode_words_to_attr(mapper_fixed.encode_attr(
                variable_el[fixed_cols]))  # encode and decode data of fixed source to ensure the same format
            self.mapper = Mapper(not_fixed_cols, self.split_expression)

            if self.add_before_perturbation is not None or self.add_after_perturbation is not None:
                self.compute_tokens(variable_el)
                if self.add_before_perturbation is not None:
                    self.add_tokens(variable_el, variable_cols, self.add_before_perturbation, self.overlap)
            variable_data = Mapper(variable_cols, self.split_expression).encode_attr(variable_el)

        elif self.variable_side == 'all':
            variable_cols = self.left_cols + self.right_cols

            self.mapper = Mapper(variable_cols, self.split_expression)
            self.fixed_data = None
            self.fixed_side = 'all'
            variable_data = self.mapper.encode_attr(variable_el)
        else:
            assert False, f'Not a feasible configuration. self.variable_side: {self.variable_side} not allowed.'
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

    def restructure_and_predict(self, perturbed_strings):
        """
            Restructure the perturbed strings from LIME and return the related predictions.
        """
        self.tmp_dataset = self.restructure_strings(perturbed_strings)
        self.tmp_dataset.reset_index(inplace=True, drop=True)
        predictions = self.model_predict(self.tmp_dataset)

        ret = np.ndarray(shape=(len(predictions), 2))
        ret[:, 1] = np.array(predictions)
        ret[:, 0] = 1 - ret[:, 1]
        return ret

    def restructure_strings(self, perturbed_strings):
        """

        Decode :param perturbed_strings into DataFrame and
        :return reconstructed pairs appending the landmark entity.

        """
        df_list = []
        for single_row in perturbed_strings:
            df_list.append(self.mapper.decode_words_to_attr_dict(single_row))
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

    def plot(self, explanation, el, figsize=(16,6)):
        exp_double = self.double_explanation_conversion(explanation, el)
        return PlotExplanation.plot(exp_double, figsize)


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
        word_dict = dict()
        res_list = list()

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
