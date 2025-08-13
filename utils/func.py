from glob import glob
import os

import numpy as np

from multiprocessing import Pool

import pickle as pkl
import warnings; warnings.filterwarnings('ignore')

import cv2
from ordpy import complexity_entropy
from scipy.stats import skew
from tqdm import tqdm



def finished(img_paths):
    for _pid in np.unique(list(map(lambda x:x.split(' - ')[-2], img_paths))):
        if not os.path.isdir(os.path.join(_save_dir, _pid)):
            return False
    return True


def filter_by_error(img):
    '''
    input: the image called cv2.imread
    output: boolean (Ture | False)
    '''
    ret = False
    if img is None:
        return True
    if img.shape[0]<=1 or img.shape[1]<=1:
        ret = True
    return ret


def get_img_features(_img_path):
    tokens = _img_path.split(' - ')
        
    pid, num_img = tokens[-2], tokens[-1].split('.')[0]
    ori_img = cv2.imread(_img_path)
    
    if not filter_by_error(ori_img):
        # convert img & get grid-like tensor
        rgb_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        _r, _g, _b = (rgb_img[:,:,0]/255).flatten(), (rgb_img[:,:,1]/255).flatten(), (rgb_img[:,:,2]/255).flatten()

        # extract features
        info_dict = {
            'img_r_mean':np.mean(_r),
            'img_r_median':np.median(_r),
            'img_r_std':np.std(_r),
            'img_g_mean':np.mean(_g),
            'img_g_median':np.median(_g),
            'img_g_std':np.std(_g),
            'img_b_mean':np.mean(_b),
            'img_b_median':np.median(_b),
            'img_b_std':np.std(_b),
        }

        # save features
        os.makedirs(os.path.join(_save_dir, pid), exist_ok=True)
        with open(os.path.join(_save_dir, pid, f'{num_img}.dict'), 'wb') as file:
            pkl.dump(info_dict, file)
        del rgb_img; del hsv_img;
    else:
        pass
    del ori_img;


def cvt_str2list(_str):
    return _str.replace('\'', '').replace('\"', '').replace('[', '').replace(']', '').replace(' ', '').split(',')


def get_unique_categories(tar_df, column):
    types_cf = []
    for combi in list(
        map(lambda x: cvt_str2list(x), tar_df[column].unique())
    ): 
        types_cf.extend(combi)
    return list(set(types_cf))


def get_group_identity(matrix):
    group_ids = np.zeros(shape=matrix.shape)
    
    init_id, switch = 1, False
    for i, s in enumerate(matrix[0,:]):
        if s==0:
            group_ids[:,i] = 0
            if switch:
                switch = False
                init_id += 1
        else:
            group_ids[:,i] = init_id
            if not switch:
                switch = True
                
    return group_ids[0,:]


def get_categorical_hue(_values):
    ret_values = _values.copy()
    _range = [i*45 for i in range(9)]
    _from, _to = _range[:-1], _range[1:]
    
    stack_indices = []
    for _st, _ed in zip(_from, _to):
        stack_indices.append(
            list(set(np.where(_st <= _values)[0]).intersection(set(np.where(_values < _ed)[0])))
        )
        
    for idx, _list in enumerate(stack_indices):
        ret_values[_list] = idx
    
    return ret_values



def get_age(curr_df):
    pub_dates = pds.to_datetime(
        curr_df[['publish_year', 'publish_month', 'publish_day']].rename(
            columns={'publish_year': 'year', 'publish_month': 'month','publish_day': 'day'}
        )
    )
    mem_dates = pds.to_datetime(
        curr_df[['membership_year', 'membership_month', 'membership_day']].rename(
            columns={'membership_year': 'year','membership_month': 'month','membership_day': 'day'}
        )
    )
    
    return (pub_dates - mem_dates).dt.days


def get_collaborator(_ref_df, _curr_df, window_size=3):
    def cvt_str2list(s):
        try:
            return eval(s) if isinstance(s, str) else s
        except:
            return []

    owner_series = _ref_df['_owner_list'].apply(cvt_str2list)
    collaborators = []

    for i in range(len(_curr_df)):
        curr_pid = _curr_df['_project_id'].iloc[i]
        ref_idx = _ref_df[_ref_df['_project_id'] == curr_pid].index[0]

        owner_window = owner_series.iloc[ref_idx - window_size : ref_idx]
        flat_owners = []
        for _owners in owner_window.values:
            flat_owners += _owners

        collaborators.append(
            np.unique(flat_owners).shape[0] - 1 
        )

    return collaborators


def compute_topic_entropy(_ref_df, _curr_df, window_size=3):
    def cvt_str2list(s):
        try:
            return eval(s) if isinstance(s, str) else s
        except:
            return []

    topics_series = _ref_df['_creative_fields'].apply(cvt_str2list)
    entropies = []

    for i in range(len(_curr_df)):
        curr_pid = _curr_df['_project_id'].iloc[i]
        ref_idx = _ref_df[_ref_df['_project_id'] == curr_pid].index[0]

        topic_window = topics_series.iloc[ref_idx - window_size : ref_idx]
        flat_topics = ['|'.join(sorted(t)) for t in topic_window if t]
        total = len(flat_topics)

        if total == 0:
            entropies.append(0)
            continue

        topic_counts = Counter(flat_topics)
        H = sum(- (count / total) * np.log(count / total) for count in topic_counts.values())
        H_norm = H / np.log(total) if total > 1 else 0
        entropies.append(H_norm)

    return entropies



def get_past_project_curation(_ref_df, _curr_df):
    past_curation_values = []

    for pid in _curr_df['_project_id']:
        idx = _ref_df[_ref_df['_project_id'] == pid].index[0]
        if idx == 0:
            past_curation_values.append(0)
        else:
            past_curation = _ref_df['_curation'].iloc[idx - 1]
            past_curation_values.append(past_curation)

    return past_curation_values


def get_recency(_ref_df, _curr_df):
    _ref_df['_publish_date'] = pds.to_datetime(_ref_df['_publish_date'])

    recency_days = []

    for pid in _curr_df['_project_id']:
        idx = _ref_df[_ref_df['_project_id'] == pid].index[0]
        if idx == 0:
            recency_days.append(None)
        else:
            prev_date = _ref_df['_publish_date'].iloc[idx - 1]
            curr_date = _ref_df['_publish_date'].iloc[idx]
            delta_days = (curr_date - prev_date).days
            recency_days.append(delta_days)

    return recency_days


def compute_cosine_similarity(_ref_df, _curr_df, window_size=3, mode='Anchored'):
    assert mode in ["Anchored", "Neighboring"], f'Not implemented for mode {mode}'

    similarities = []

    for i in range(len(_curr_df)):
        curr_pid = _curr_df['_project_id'].iloc[i]
        ref_idx = _ref_df[_ref_df['_project_id'] == curr_pid].index[0]


        window = _ref_df.iloc[ref_idx - window_size : ref_idx]
        
        vectors = window[[f'Artwork_Visual_Features_latent_PCA_embeddings_dim-{j}' for j in range(1, 201)]].to_numpy()

        if len(vectors) < 2:
            similarities.append(np.nan)
            continue

        if mode == 'Anchored':
            anchor_vec = vectors[-1]
            sims = [1 - cosine(anchor_vec, vectors[j]) for j in range(len(vectors) - 1)]
        elif mode == 'Neighboring':
            sims = [1 - cosine(vectors[j], vectors[j+1]) for j in range(len(vectors) - 1)]

        avg_sim = np.mean(sims)
        similarities.append(avg_sim)

    return similarities


def compute_euclidean_distance(_ref_df, _curr_df, what, window_size=3, mode='Anchored'):
    assert mode in ["Anchored", "Neighboring"], f'Not implemented for mode {mode}'

    if what == 'hsv':
        features = [
            'Artwork_Visual_Features_hue_Mean',
            'Artwork_Visual_Features_saturation_Mean',
            'Artwork_Visual_Features_value_Mean'
        ]
    elif what == 'rgb':
        features = [
            'Artwork_Visual_Features_red_Mean',
            'Artwork_Visual_Features_green_Mean',
            'Artwork_Visual_Features_blue_Mean',
        ]
    elif what == 'ch':
        features = [
            'Artwork_Visual_Features_Complexity',
            'Artwork_Visual_Features_Entropy'
        ]

    distances = []

    for i in range(len(_curr_df)):
        curr_pid = _curr_df['_project_id'].iloc[i]
        ref_idx = _ref_df[_ref_df['_project_id'] == curr_pid].index[0]

        window = _ref_df.iloc[ref_idx - window_size : ref_idx]
        vectors = window[features].to_numpy()

        if len(vectors) < 2:  
            distances.append(np.nan)
            continue

        if mode == 'Anchored':
            anchor_vec = vectors[-1]
            dists = [euclidean(anchor_vec, vectors[j]) for j in range(len(vectors) - 1)]
        elif mode == 'Neighboring':
            dists = [euclidean(vectors[j], vectors[j + 1]) for j in range(len(vectors) - 1)]

        avg_dist = np.mean(dists)
        distances.append(avg_dist)


    return distances
