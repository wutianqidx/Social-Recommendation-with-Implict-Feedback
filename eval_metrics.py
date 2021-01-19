import numpy as np
import random

''' M is # relevant outputs'''

def recall_at_k(relevance_score, K, M):
    assert K >= 1
    relevance_score = np.asarray(relevance_score)[:K] != 0
    if relevance_score.size != K:
        raise ValueError('Relevance score length < K')
    return float(np.sum(relevance_score)) / min(K, float(M))


def mean_recall_at_k(relevance_scores, K, M_list):
    mean_recall_at_k = np.mean([recall_at_k(r, K, M) for r, M in zip(relevance_scores, M_list)]).astype(np.float32)
    return mean_recall_at_k


def DCG_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.


def NDCG_at_k(r, k):
    dcg_max = DCG_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return DCG_at_k(r, k) / dcg_max


def mean_NDCG_at_k(relevance_scores, k, M_list):
    mean_ndcg = np.mean([NDCG_at_k(r, k) for r in relevance_scores]).astype(np.float32)
    return mean_ndcg


def get_relevance_scores_bpr(rec_list, actual_list):
    # Assume inputs as [U, # rec items] or [U, # true items]: (list of lists) or (dict of lists)
    # Return a binary relevance score for each U over all items?
    output = []
    M_list = []
    for user in rec_list:
        rec_item_list = rec_list[user]
        z = np.isin(rec_item_list, list(actual_list[user].keys()))
        output.append(z)
        M_list.append(len(actual_list[user].keys()))
    return np.array(output), M_list

metrics = {}
metrics['Recall'] = mean_recall_at_k
metrics['NDCG'] = mean_NDCG_at_k

# rec_list gives the top-k recommended items among all items.

def ranking_metrics(rec_list, actual_list, k_list=[50], mode='test', method='recq'):
    rows = []
    relevances_scores, M_list = get_relevance_scores_bpr(rec_list, actual_list)
    for k in k_list:
        for metric, metric_fn in metrics.items():
            row_dict = {}
            row_dict["Metric"] = metric
            row_dict["K"] = k
            row_dict["mode"] = mode
            row_dict["score"] = metric_fn(relevances_scores, k, M_list)
            print ("{}@{} : {}".format(metric, k, row_dict["score"]))
            rows.append(row_dict)
    return rows
