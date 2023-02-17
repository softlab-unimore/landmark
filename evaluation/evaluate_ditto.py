from notebook_import_utility_env import *
from functools import partial
from tqdm.autonotebook import tqdm
import pandas as pd
import os
import sys



sorted_dataset_names = [
    'BeerAdvo-RateBeer',
    'fodors-zagats',
    'iTunes-Amazon',
    'dirty_itunes_amazon',
    'DBLP-Scholar',
    'dirty_dblp_scholar',
    'walmart-amazon',
    'dirty_walmart_amazon',
    'DBLP-ACM',
    'dirty_dblp_acm',
    'Amazon-Google',
    'Abt-Buy',
]
tasks = [
    'Structured/Beer',
    'Structured/Fodors-Zagats',
    'Structured/iTunes-Amazon',
    'Dirty/iTunes-Amazon',
    'Structured/DBLP-GoogleScholar',
    'Dirty/DBLP-GoogleScholar',
    'Structured/Walmart-Amazon',
    'Dirty/Walmart-Amazon',
    'Structured/DBLP-ACM',
    'Dirty/DBLP-ACM',
    'Structured/Amazon-Google',
    'Textual/Abt-Buy', ]



def compute_explanations(mirror_elements=False, redo=False, num_explanations=100):

    from wrapper.DITTOWrapper import DITTOWrapper
    from landmark import Landmark



def compute_explanations(mirror_elements=False, redo=False, num_explanations=100):

    from wrapper.DITTOWrapper import DITTOWrapper
    from landmark import Landmark

    checkpoint_path = os.path.join(
        os.path.join(softlab_path, 'Projects/external_github/ditto/checkpoints'))  # 'checkpoints'

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

        model = DITTOWrapper(task, checkpoint_path)
        test_df = dataset_dict['test']
        explainer = Landmark(partial(model.predict, batch_size=batch_size), test_df,
                             exclude_attrs=exclude_attrs + ['label', 'id'], lprefix='left_', rprefix='right_',
                             split_expression=r' ')

        turn_df = test_df
        pos_mask = turn_df['label'] == 1
        pos_df = turn_df[pos_mask]
        neg_df = turn_df[~pos_mask]
        pos_sample = pos_df.sample(num_explanations, random_state=0) if pos_df.shape[0] >= num_explanations else pos_df
        neg_sample = neg_df.sample(num_explanations, random_state=0) if neg_df.shape[0] >= num_explanations else neg_df

        if mirror_elements is False:
            for conf in ['single', 'double']:
                for sample, prefix in zip([pos_sample, neg_sample], ['positive', 'negative']):
                    tmp_path = os.path.join(turn_files_path, f'{prefix}_explanations_{conf}.csv')
                    print(f'{prefix} explanations')
                    try:
                        assert redo is False, 'redo was set.'
                        tmp_df = pd.read_csv(tmp_path)
                        assert tmp_df.id.nunique() >= sample.shape[0], 'Not computed'
                        print('loaded')
                    except Exception as e:
                        print(e)
                        tmp_df = explainer.explain(sample, num_samples=num_samples, conf=conf)
                        tmp_df.to_csv(tmp_path, index=False)

        else:
            for conf in ['single']:
                for sample, prefix in zip([pos_sample, neg_sample], ['positive', 'negative']):
                    tmp_path = os.path.join(turn_files_path, f'{prefix}_explanations_{conf}.csv')
                    print(f'{prefix} explanations')
                    try:
                        tmp_df = pd.read_csv(tmp_path)
                        assert tmp_df.id.nunique() >= sample.shape[0], 'Not computed'
                        # assert False
                        print('loaded')
                    except Exception as e:
                        print(e)
                        tmp_df = explainer.explain(sample, num_samples=num_samples, conf=conf)
                        tmp_df.to_csv(tmp_path, index=False)



def load_explanations_DITTO(softlab_path, num_explanations=100, mirror_elements=False):
    if mirror_elements:
        num_explanations = num_explanations // 2

    checkpoint_path = os.path.join(
        os.path.join(softlab_path, 'Projects/external_github/ditto/checkpoints'))  # 'checkpoints'
    res = {}
    for i in tqdm(range(len(sorted_dataset_names))):
        turn_dict = {}

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
        turn_dict.update(**dataset_dict)

        test_df = dataset_dict['test']
        turn_df = test_df
        pos_mask = turn_df['label'] == 1
        pos_df = turn_df[pos_mask]
        neg_df = turn_df[~pos_mask]
        pos_sample = pos_df.sample(num_explanations, random_state=0) if pos_df.shape[0] >= num_explanations else pos_df
        neg_sample = neg_df.sample(num_explanations, random_state=0) if neg_df.shape[0] >= num_explanations else neg_df

        turn_dict['pos_sample'] = pos_sample
        turn_dict['neg_sample'] = neg_sample

        if mirror_elements:
            for conf in ['single']:
                for sample, prefix in zip([pos_sample, neg_sample], ['positive', 'negative']):
                    tmp_path = os.path.join(turn_files_path, f'{prefix}_explanations_{conf}_mirror.csv')
                    # '/home/baraldian/content/drive/Shareddrives/SoftLab/Projects/Landmark Explanation EM/dataset_files/BeerAdvo-RateBeer/positive_explanations_single_mirror.csv'
                    print(f'{prefix} explanations')
                    try:
                        tmp_df = pd.read_csv(tmp_path, index_col='index')
                        turn_dict[f'{prefix}_explanations_{conf}_mirror'] = tmp_df
                        assert tmp_df.id.nunique() >= sample.shape[0], 'Not computed'
                        print('loaded')
                    except Exception as e:
                        print(e)
                        sys.path.append(os.path.join(softlab_path, 'Projects/Landmark Explanation EM/Landmark_github'))
                        sys.path.append(os.path.join(softlab_path, 'Projects/external_github/ditto'))
                        sys.path.append(os.path.join(softlab_path, 'Projects/external_github'))
                        from wrapper.DITTOWrapper import DITTOWrapper
                        from landmark import Landmark
                        batch_size = 2048
                        num_samples = 2048
                        model = DITTOWrapper(task, checkpoint_path)
                        test_df = dataset_dict['test']
                        explainer = Landmark(partial(model.predict, batch_size=batch_size), test_df,
                                             exclude_attrs=exclude_attrs + ['label', 'id'], lprefix='left_',
                                             rprefix='right_', split_expression=r' ')
                        # Mirror df and explain
                        explanation_list = []
                        for side_to_copy in ['left_', 'right_']:
                            side_to_write = 'left_' if side_to_copy == 'right_' else 'right_'
                            turn_sample = sample.copy()
                            for col_to_copy in sample.columns[sample.columns.str.startswith(side_to_copy)]:
                                col_name = col_to_copy.replace(side_to_copy, '')
                                turn_sample[side_to_write + col_name] = turn_sample[col_to_copy]
                            tmp_df = explainer.explain(turn_sample, num_samples=num_samples, conf=conf)
                            tmp_df['mirror_side'] = side_to_copy
                            explanation_list.append(tmp_df)
                        exp_df = pd.concat(explanation_list)
                        exp_df.to_csv(tmp_path, index=False)
                        turn_dict[f'{prefix}_explanations_{conf}_mirror'] = exp_df
        else:
            for conf in ['single', 'double']:
                for sample, prefix in zip([pos_sample, neg_sample], ['positive', 'negative']):
                    tmp_path = os.path.join(turn_files_path, f'{prefix}_explanations_{conf}.csv')
                    print(f'{prefix} explanations')
                    try:
                        # assert False
                        tmp_df = pd.read_csv(tmp_path, index_col='index')
                        turn_dict[f'{prefix}_explanations_{conf}'] = tmp_df
                        assert tmp_df.id.nunique() >= sample.shape[0], 'Not computed'
                        # assert False
                        print('loaded')
                    except Exception as e:
                        print(e)
        res[turn_dataset_name] = turn_dict
    return res


base_files_path = os.path.join(softlab_path, 'Projects/Landmark Explanation EM/dataset_files')


def load_explanations_DITTO(softlab_path, num_explanations=100, mirror_elements=False):
    if mirror_elements:
        num_explanations = num_explanations // 2

    checkpoint_path = os.path.join(
        os.path.join(softlab_path, 'Projects/external_github/ditto/checkpoints'))  # 'checkpoints'
    res = {}
    for i in tqdm(range(len(sorted_dataset_names))):
        turn_dict = {}

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
        turn_dict.update(**dataset_dict)

        test_df = dataset_dict['test']
        turn_df = test_df
        pos_mask = turn_df['label'] == 1
        pos_df = turn_df[pos_mask]
        neg_df = turn_df[~pos_mask]
        pos_sample = pos_df.sample(num_explanations, random_state=0) if pos_df.shape[0] >= num_explanations else pos_df
        neg_sample = neg_df.sample(num_explanations, random_state=0) if neg_df.shape[0] >= num_explanations else neg_df

        turn_dict['pos_sample'] = pos_sample
        turn_dict['neg_sample'] = neg_sample

        if mirror_elements:
            for conf in ['single']:
                for sample, prefix in zip([pos_sample, neg_sample], ['positive', 'negative']):
                    tmp_path = os.path.join(turn_files_path, f'{prefix}_explanations_{conf}_mirror.csv')
                    # '/home/baraldian/content/drive/Shareddrives/SoftLab/Projects/Landmark Explanation EM/dataset_files/BeerAdvo-RateBeer/positive_explanations_single_mirror.csv'
                    print(f'{prefix} explanations')
                    try:
                        tmp_df = pd.read_csv(tmp_path, index_col='index')
                        turn_dict[f'{prefix}_explanations_{conf}_mirror'] = tmp_df
                        assert tmp_df.id.nunique() >= sample.shape[0], 'Not computed'
                        print('loaded')
                    except Exception as e:
                        print(e)
                        sys.path.append(os.path.join(softlab_path, 'Projects/Landmark Explanation EM/Landmark_github'))
                        sys.path.append(os.path.join(softlab_path, 'Projects/external_github/ditto'))
                        sys.path.append(os.path.join(softlab_path, 'Projects/external_github'))
                        from wrapper.DITTOWrapper import DITTOWrapper
                        from landmark import Landmark
                        batch_size = 2048
                        num_samples = 2048
                        model = DITTOWrapper(task, checkpoint_path)
                        test_df = dataset_dict['test']
                        explainer = Landmark(partial(model.predict, batch_size=batch_size), test_df,
                                             exclude_attrs=excluded_cols + ['label', 'id'], lprefix='left_',
                                             rprefix='right_', split_expression=r' ')
                        # Mirror df and explain
                        explanation_list = []
                        for side_to_copy in ['left_', 'right_']:
                            side_to_write = 'left_' if side_to_copy == 'right_' else 'right_'
                            turn_sample = sample.copy()
                            for col_to_copy in sample.columns[sample.columns.str.startswith(side_to_copy)]:
                                col_name = col_to_copy.replace(side_to_copy, '')
                                turn_sample[side_to_write + col_name] = turn_sample[col_to_copy]
                            tmp_df = explainer.explain(turn_sample, num_samples=num_samples, conf=conf)
                            tmp_df['mirror_side'] = side_to_copy
                            explanation_list.append(tmp_df)
                        exp_df = pd.concat(explanation_list)
                        exp_df.to_csv(tmp_path, index=False)
                        turn_dict[f'{prefix}_explanations_{conf}_mirror'] = exp_df
        else:
            for conf in ['single', 'double']:
                for sample, prefix in zip([pos_sample, neg_sample], ['positive', 'negative']):
                    tmp_path = os.path.join(turn_files_path, f'{prefix}_explanations_{conf}.csv')
                    print(f'{prefix} explanations')
                    try:
                        # assert False
                        tmp_df = pd.read_csv(tmp_path, index_col='index')
                        turn_dict[f'{prefix}_explanations_{conf}'] = tmp_df
                        assert tmp_df.id.nunique() >= sample.shape[0], 'Not computed'
                        # assert False
                        print('loaded')
                    except Exception as e:
                        print(e)
        res[turn_dataset_name] = turn_dict
    return res


base_files_path = os.path.join(softlab_path, 'Projects/Landmark Explanation EM/dataset_files')

if __name__ == "__main__":
    os.chdir(os.path.join(softlab_path, 'Projects/external_github/ditto'))
    # load_explanations_DITTO(softlab_path)
    load_explanations_DITTO(softlab_path, num_explanations=30, mirror_elements=True)
    # compute_explanations(redo=True, num_explanations=2)
