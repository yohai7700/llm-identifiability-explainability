import torch
from torch.utils.data import Subset

from transformers import TrainingArguments, Trainer, logging, DataCollatorWithPadding
from args import get_args
from models.classification_model import model, tokenizer, data_collator
from models.lora import attach_lora
from training.evaluation_metrics import compute_metrics

from data.list_dataset import ListDataset
from data.utils.preprocessing import get_preprocessed_dataset_path, get_model_alias

learning_rate = 1e-3
batch_size = 8

def get_classification_model_folder(training_llm = get_args().llm_generating_model_name):
    dataset_name = get_args().source_dataset_type
    if get_args().include_training_subset_size_in_classifier_folder:
        dataset_name += f"_{get_args().training_subset_size}"
    return f"models/checkpoints/{get_args().classification_model_name}_{dataset_name}_{get_model_alias(training_llm)}"

training_args = TrainingArguments(
    output_dir=get_classification_model_folder(),
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=get_args().epochs,
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

train_dataset = Subset(ListDataset(torch.load(get_preprocessed_dataset_path('train'), weights_only=True)), range(get_args().training_subset_size))
eval_dataset = Subset(ListDataset(torch.load(get_preprocessed_dataset_path('eval'), weights_only=True)), range(get_args().eval_subset_size))

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