import torch
import numpy as np
from sklearn.neighbors  import (
    NearestNeighbors,
    LocalOutlierFactor
)
from sklearn.metrics import DistanceMetric


def maha_scores(embs_test, embs_train):
    mean_emb_per_sec = np.mean(embs_train, axis=0)
    cov_per_sec = np.cov(embs_train, rowvar=False)
    if np.isnan(cov_per_sec).sum() > 0:
        raise ValueError("there is nan in the cov of train_embs")
    cov_per_sec += 1e-6 * np.eye(cov_per_sec.shape[0])
    dist = DistanceMetric.get_metric('mahalanobis', V=cov_per_sec)
    scores = dist.pairwise([mean_emb_per_sec], embs_test)[0]
    # scores of train data
    scores_train = dist.pairwise([mean_emb_per_sec], embs_train)[0]
    return scores, scores_train

def knn_scores(embs_test, embs_train):
    clf = NearestNeighbors(
        n_neighbors=2
        # metric='mahalanobis'
        # metric='cosine'
    )
    clf.fit(embs_train)
    scores = clf.kneighbors(embs_test)[0].sum(-1)
    scores_train = None
    return scores, scores_train

def lof_scores(embs_test, embs_train):
    lof = LocalOutlierFactor(
        n_neighbors=4,
        contamination=1e-6,
        metric='cosine',
        novelty=True,
    )
    lof.fit(embs_train)
    scores_train = -lof.negative_outlier_factor_
    scores = -lof.score_samples(embs_test)
    return scores, scores_train

def cos_scores(embs_test, embs_train):
    mean_emb_per_sec = torch.from_numpy(
        np.mean(embs_train, axis=0)
    ).repeat(embs_test.shape[0], 1)
    embs_test = torch.from_numpy(embs_test)
    scores = 1 - torch.cosine_similarity(
        embs_test, mean_emb_per_sec
    ).numpy()
    embs_train = torch.from_numpy(embs_train)
    # scores_train = 1 - torch.cosine_similarity(
    #     embs_train, mean_emb_per_sec
    # ).numpy()
    return scores, None 
