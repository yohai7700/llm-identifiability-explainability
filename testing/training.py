from transformers import TrainingArguments, Trainer
from testing.model import tokenizer, data_collator, tokenize_function, tokenized_dataset
from testing.lora import lora_model
from testing.evaluation import compute_metrics
from testing.dataset import dataset
from testing.artificial_llm_text_dataset import ArtificialLlmTextDataset

learning_rate = 1e-3
batch_size = 4
num_epochs = 10

training_args = TrainingArguments(
    output_dir = "text-classification",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

train_dataset = ArtificialLlmTextDataset(dataset['train'])
eval_dataset = ArtificialLlmTextDataset(dataset['validation'])

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    # train_dataset=tokenized_dataset['train'],
    # eval_dataset=tokenized_dataset['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)