import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from args import get_args

def predict():
    model = AutoModelForSequenceClassification.from_pretrained('./models/checkpoints/llm_cls/distilbert_qwen/model',device_map='cuda')
    tokenizer = AutoTokenizer.from_pretrained('./models/checkpoints/llm_cls/distilbert_qwen/model', device_map='cuda')

    text = get_args().prediction_text_input
    if text is None:
        text = input("Enter a text to predict: ")

    inputs = tokenizer(text, return_tensors="pt").to('cuda')
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits

    prediction = logits.argmax().item()
    return prediction