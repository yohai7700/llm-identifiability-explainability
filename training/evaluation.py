import torch
from transformers import AutoModelForSequenceClassification, Trainer

from data.list_dataset import ListDataset
from training.evaluation_metrics import compute_metrics
from models.classification_model import data_collator
from training.trainer import get_classification_model_folder
from data.utils.preprocessing import get_preprocessed_dataset_path


def eval():
    model_folder = get_classification_model_folder()
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
