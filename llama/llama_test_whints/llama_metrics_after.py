import json
import pickle
import numpy as np

def mean_accuracy(y_true, y_pred):
    return 1 - np.mean(np.abs(y_true - y_pred))

def concordance_correlation_coefficient(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    return (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2)

def load_ground_truth(path):
    with open(path, 'rb') as f:
        raw = pickle.load(f, encoding='latin1')
    traits = list(raw.keys())
    clip_ids = sorted(raw[traits[0]].keys())
    return {
        clip: {trait: float(raw[trait][clip]) for trait in traits}
        for clip in clip_ids
    }

def load_predictions(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_metrics(gt_dict, pred_dict):
    clips = sorted(set(gt_dict) & set(pred_dict))
    traits = sorted(gt_dict[clips[0]].keys())
    y_t, y_p = [], []
    for clip in clips:
        for trait in traits:
            y_t.append(gt_dict[clip][trait])
            y_p.append(float(pred_dict[clip].get(trait, np.nan)))
    y_t = np.array(y_t)
    y_p = np.array(y_p)
    return mean_accuracy(y_t, y_p), concordance_correlation_coefficient(y_t, y_p)

if __name__ == '__main__':
    gt_path = '../annotation_test.pkl'
    zero_pred_path = 'llama25_zero_with_hints.json'
    one_pred_path = 'llama25_one_with_hints.json'

    gt = load_ground_truth(gt_path)
    zero_preds = load_predictions(zero_pred_path)
    one_preds = load_predictions(one_pred_path)

    z_acc, z_ccc = compute_metrics(gt, zero_preds)
    o_acc, o_ccc = compute_metrics(gt, one_preds)

    print(f"Zero-shot custom: mAcc = {z_acc:.4f}, CCC = {z_ccc:.4f}")
    print(f"One-shot custom:  mAcc = {o_acc:.4f}, CCC = {o_ccc:.4f}")