import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")

def compute_metrics(examples):
    predictions, labels = examples
    predictions = np.argmax(predictions, axis=1)

    accuracy_value = accuracy.compute(predictions=predictions, references=labels)
    return { 'accuracy': accuracy_value }