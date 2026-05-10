from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import numpy as np

def compute_metrics(y_true, y_pred):
return {
"roc_auc": roc_auc_score(y_true, y_pred),
"pr_auc": average_precision_score(y_true, y_pred),
"f1": f1_score(y_true, (y_pred > 0.5).astype(int))
}
