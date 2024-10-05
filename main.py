import torch

import transformers
import accelerate
import peft
from peft import LoraConfig, get_peft_model, TaskType

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
import evaluate


print(f"Transformers version: {transformers.__version__}")
print(f"Accelerate version: {accelerate.__version__}")
print(f"PEFT version: {peft.__version__}")
print(torch.cuda.is_available())

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

dataset = load_dataset("yelp_review_full")

device = "cuda" if torch.cuda.is_available() else "cpu"

training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="steps", eval_steps=5, per_device_train_batch_size=4, num_train_epochs=5)

# model = AutoModelForCausalLM.from_pretrained("gpt2-medium", device_map=device)
classification_model = AutoModelForSequenceClassification.from_pretrained("gpt2-medium", device_map=device, num_labels=5)
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token

prompt = "My favourite condiment is"

model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
# model.to(device)
classification_model.to(device)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# generated_ids = model.generate(**model_inputs, max_new_tokens=20, do_sample=True)
# print(tokenizer.batch_decode(generated_ids)[0])
# result = classification_model(**model_inputs)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["mlp"],
)
lora_model = get_peft_model(classification_model, lora_config)
# print_trainable_parameters(lora_model)

lora_model.config.pad_token_id = lora_model.config.eos_token_id

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()