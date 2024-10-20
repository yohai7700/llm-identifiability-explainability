import torch

from transformers import TrainingArguments, Trainer, logging
from model import model, tokenizer, data_collator
from lora import attach_lora
from evaluation import compute_metrics
from dataset import train_dataset, eval_dataset
from list_dataset import ListDataset

learning_rate = 1e-3
batch_size = 4
num_epochs = 10

training_args = TrainingArguments(
    output_dir = "model_checkpoints/llm_classification",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

train_dataset = ListDataset(torch.load('./dataset_checkpoints/yelp/train_dataset.pt'))
eval_dataset = ListDataset(torch.load('./dataset_checkpoints/yelp/eval_dataset.pt'))

lora_model = attach_lora(model, tokenizer)
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
logging.set_verbosity_warning()