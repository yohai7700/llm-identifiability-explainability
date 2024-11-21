import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoModelForSequenceClassification, Trainer, pipeline, Pipeline

from data.list_dataset import ListDataset
from training.evaluation_metrics import compute_metrics
from models.classification_model import data_collator
from training.trainer import get_classification_model_folder
from data.utils.preprocessing import get_preprocessed_dataset_path
from args import get_args


def eval():
    if get_args().is_baseline:
        return evaluate_baseline()

    model_folder = get_classification_model_folder(get_args().training_llm_generating_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(f'{model_folder}/model',device_map='cuda')

    eval_dataset_path = get_preprocessed_dataset_path('eval')
    print(f"Model Path: {model_folder}")
    print(f"Evaluation Dataset Path: {eval_dataset_path}")
    
    eval_dataset = ListDataset(torch.load(eval_dataset_path, weights_only=True))
    trainer = Trainer(
        model=model, 
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    evaluation = trainer.evaluate()
    print(evaluation)

def evaluate_baseline():
    pipe = pipeline("text-classification", model="roberta-base-openai-detector", device_map="auto")
    eval_dataset_path = get_preprocessed_dataset_path('eval')
    print(f"Evaluation Dataset Path: {eval_dataset_path}")
    
    eval_dataset = ListDataset(torch.load(eval_dataset_path, weights_only=True))
    print(evaluate_metrics_over_dataset(pipe, eval_dataset, compute_metrics))

def evaluate_metrics_over_dataset(pipe: Pipeline, dataset, compute_metrics, max_length=512, fake_label="Fake", real_label="Real"):
    """
    Evaluates the pipeline on a dataset and computes metrics.
    
    Parameters:
        pipe: Hugging Face pipeline for text classification.
        dataset: List of dictionaries, each containing 'text' and 'label'.
        compute_metrics: Function to compute metrics, expects a tuple of predictions and labels.
    
    Returns:
        Dictionary containing evaluation metrics.
    """
    prediction_scores = []
    labels = []

    # Iterate through the dataset
    for example in tqdm(dataset, desc="Evaluating"):
        text = example['text']
        label = example['label']
        labels.append(label)

        # Get pipeline output
        prediction = pipe(text, top_k=None, truncation=True, max_length=max_length)
        if len(prediction) is not 2:
            raise "Pipeline output is not as expected."
        if prediction[0]['label'] == real_label and prediction[1]['label'] == fake_label:
            scores = [prediction[0]['score'], prediction[1]['score']]
        elif prediction[0]['label'] == fake_label and prediction[1]['label'] == real_label:
            scores = [prediction[1]['score'], prediction[0]['score']]
        else:
            raise "Pipeline output is not as expected."
        prediction_scores.append(scores)

    # Convert to numpy arrays for metric calculation
    prediction_scores = np.array(prediction_scores)
    labels = np.array(labels)

    # Compute and return metrics
    return compute_metrics((prediction_scores, labels))