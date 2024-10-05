from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from testing.dataset import dataset

model_name = 'distilbert-base-uncased'

id2label = {0: 'Negative', 1: 'Positive'}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2, 
    id2label=id2label, 
    label2id=label2id
)
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

def tokenize_function(example):
    text = example['text']

    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=512
    )

    return tokenized_inputs

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

tokenized_dataset = dataset.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)