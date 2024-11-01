import evaluate
import numpy as np

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
confusion_matrix_metric = evaluate.load("confusion_matrix")

def compute_metrics(examples):
    predictions, labels = examples
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1_score = f1_metric.compute(predictions=predictions, references=labels)
    confusion_matrix = confusion_matrix_metric.compute(predictions=predictions, references=labels)
    confusion_matrix['confusion_matrix'] = confusion_matrix['confusion_matrix'].tolist()
    return { 'accuracy': accuracy, 'f1_score': f1_score, 'confusion_matrix': confusion_matrix }