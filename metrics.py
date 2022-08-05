import torch

def exact_match(y_true, y_pred, **kwargs):
    with torch.no_grad():
        return (y_true == y_pred).all(dim=1).mean().item()

def overlap_f1_score(y_true, y_pred, **kwargs):
  with torch.no_grad():
    y_true_start, y_true_end = y_true[:, 0], y_true[:, 1]
    y_pred_start, y_pred_end = y_pred[:, 0], y_pred[:, 1]
    tp = torch.maximum(torch.tensor(0), torch.minimum(y_true_end, y_pred_end) -\
                                        torch.maximum(y_true_start, y_pred_start) +\
                                        1)
    fn = torch.maximum(y_true_start, y_pred_start) - y_true_start +\
        y_true_end - torch.minimum(y_true_end, y_pred_end)
    fp = y_true_start - torch.minimum(y_true_start, y_pred_start) +\
        torch.maximum(y_true_end, y_pred_end) - y_true_end
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = torch.nan_to_num(2 * precision * recall / (precision + recall))
    return f1.mean().item()