import torch
from transformers import AutoModelForSequenceClassification, Trainer

from data.list_dataset import ListDataset
from training.evaluation_metrics import compute_metrics
from models.classification_model import data_collator


def eval():
    model = AutoModelForSequenceClassification.from_pretrained('./models/checkpoints/llm_cls/distilbert_qwen/model',device_map='cuda')

    eval_dataset = ListDataset(torch.load('./data/checkpoints/yelp/eval_dataset.pt', weights_only=True))
    trainer = Trainer(
        model=model, 
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    evaluation = trainer.evaluate()
    print(evaluation)
