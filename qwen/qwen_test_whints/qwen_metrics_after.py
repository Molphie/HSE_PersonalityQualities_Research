import json
import pickle
import numpy as np

def mean_accuracy(y_true, y_pred):
    return 1 - np.mean(np.abs(y_true - y_pred))

def concordance_correlation_coefficient(y_true, y_pred):
    mean_t = np.mean(y_true)
    mean_p = np.mean(y_pred)
    cov = np.mean((y_true - mean_t) * (y_pred - mean_p))
    var_t = np.var(y_true)
    var_p = np.var(y_pred)
    return (2 * cov) / (var_t + var_p + (mean_t - mean_p) ** 2)

def load_ground_truth(path):
    with open(path, 'rb') as f:
        raw = pickle.load(f, encoding='latin1')
    gt = {}
    for trait, m in raw.items():
        for clip, v in m.items():
            gt.setdefault(str(clip), {})[trait] = float(v)

def load_predictions(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    preds = {}
    for clip, v in raw.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            preds[str(clip)] = {k: float(v[0][k]) for k in v[0].keys()}
        elif isinstance(v, dict):
            preds[str(clip)] = {k: float(v[k]) for k in v.keys()}
        else:
            preds[str(clip)] = {}
    return preds

def compute_metrics(gt, preds):
    common_clips = sorted(set(gt) & set(preds))
    sample = common_clips[0]
    common_traits = sorted(set(gt[sample].keys()) & set(preds[sample].keys()))
    y_t, y_p = [], []
    for clip in common_clips:
        for trait in common_traits:
            y_t.append(gt[clip][trait])
            y_p.append(preds[clip].get(trait, np.nan))
    y_t = np.array(y_t)
    y_p = np.array(y_p)
    return mean_accuracy(y_t, y_p), concordance_correlation_coefficient(y_t, y_p)

if __name__ == '__main__':
    gt_path        = '../annotation_test.pkl'
    zero_path      = 'qwen25_zero_with_hints.json'
    one_path       = 'qwen25_one_with_hints.json'

    gt    = load_ground_truth(gt_path)
    zero  = load_predictions(zero_path)
    one   = load_predictions(one_path)

    z_acc, z_ccc = compute_metrics(gt, zero)
    o_acc, o_ccc = compute_metrics(gt, one)

    print(f"Qwen25 Zero-shot custom: mAcc = {z_acc:.4f}, CCC = {z_ccc:.4f}")
    print(f"Qwen25 One-shot custom:  mAcc = {o_acc:.4f}, CCC = {o_ccc:.4f}")