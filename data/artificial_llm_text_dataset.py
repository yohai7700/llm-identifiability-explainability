import torch

from transformers import pipeline
from args import get_args
from models.classification_model import tokenize_function_artificial

class ArtificialLlmTextDataset(torch.utils.data.Dataset):
    """Some Information about ArtificialLlmTextDataset"""
    def __init__(self, original_dataset: torch.utils.data.Dataset):
        super(ArtificialLlmTextDataset, self).__init__()

        self.original_dataset = original_dataset
        
        self.pipe = pipeline("text-generation", model=get_args().llm_generating_model_name, trust_remote_code=True, device_map="cuda")

    def __getitem__(self, index):
        text_key = self.get_text_key()
        original_text = self.original_dataset[index][text_key]
        
        if index % 2 == 0:
            text = original_text
            label = 0
        else:
            text = self.rewrite(original_text)
            label = 1

        tokenized_properties = tokenize_function_artificial({ 'text': text })
        result = { 'text': text, 'label': label, 'pre_llm_text': original_text }
        result.update(tokenized_properties)

        return result
    
    def get_text_key(self):
        if get_args().source_dataset_type == "amazon_polarity":
            return "content"
        
        return "text"
    
    def rewrite(self, text: str):
        if not text.startswith('"'):
            text = f'"{text}'
        if not text.endswith('"'):
            text = f'{text}"'

        messages = [
            {"role": "user", "content": f"rewrite the following text: {text}"},
        ]
        result = self.pipe(messages, max_length=2000)
        rewritten_text = result[0]['generated_text'][1]['content']
        return rewritten_text

    def __len__(self):
        return len(self.original_dataset)