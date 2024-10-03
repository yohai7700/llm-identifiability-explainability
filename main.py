import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
import evaluate

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

dataset = load_dataset("yelp_review_full")
dataset["train"][100]

device = "cuda" if torch.cuda.is_available() else "cpu"

training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")

model = AutoModelForCausalLM.from_pretrained("gpt2-medium", device_map=device)
classification_model = AutoModelForSequenceClassification.from_pretrained("gpt2-medium", device_map=device, num_labels=5)
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token

prompt = "My favourite condiment is"

model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
model.to(device)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# generated_ids = model.generate(**model_inputs, max_new_tokens=20, do_sample=True)
# print(tokenizer.batch_decode(generated_ids)[0])
# result = classification_model(**model_inputs)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()