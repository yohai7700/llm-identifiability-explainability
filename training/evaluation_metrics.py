import evaluate
import numpy as np
from scipy.special import softmax

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
confusion_matrix_metric = evaluate.load("confusion_matrix")
roc_auc_metric = evaluate.load("roc_auc")

def compute_metrics(examples):
    prediction_scores, labels = examples
    predictions = np.argmax(prediction_scores, axis=1)
    prediction_scores = softmax(prediction_scores, axis=1)
    prediction_scores = [score[1] for score in prediction_scores]

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)['accuracy']
    f1_score = f1_metric.compute(predictions=predictions, references=labels)['f1']
    confusion_matrix = confusion_matrix_metric.compute(predictions=predictions, references=labels)['confusion_matrix']
    confusion_matrix = confusion_matrix.tolist()

    roc_auc_score = roc_auc_metric.compute(prediction_scores=prediction_scores, references=labels)['roc_auc']
    return { 'accuracy': accuracy, 'f1_score': f1_score, 'confusion_matrix': confusion_matrix, 'roc_auc': roc_auc_score }