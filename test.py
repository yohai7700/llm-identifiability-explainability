from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

from args import get_args

model_name = get_args().classification_model_name

id2label = {0: 'Negative', 1: 'Positive'}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2, 
    id2label=id2label, 
    label2id=label2id,
    cache_dir=get_args().cache_dir,
    device_map=get_args().device
)
print(model)