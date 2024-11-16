import torch

from transformers import pipeline
from args import get_args
from models.classification_model import tokenize_function_artificial

class ArtificialLlmTextDataset(torch.utils.data.Dataset):
    """Some Information about ArtificialLlmTextDataset"""
    def __init__(self, original_dataset: torch.utils.data.Dataset, dataset_name=''):
        super(ArtificialLlmTextDataset, self).__init__()

        self.original_dataset = original_dataset
        
        self.pipe = pipeline("text-generation", model=get_args().llm_generating_model_name, trust_remote_code=False, device_map="auto")
        self.dataset_name = dataset_name

    def __getitem__(self, index):
        if self.dataset_name == 'imdb' or self.dataset_name == 'yelp':
            original_text = self.original_dataset[index]['text']
        elif self.dataset_name == 'amazon_polarity':
            original_text = self.original_dataset[index]['content']
        elif self.dataset_name == 'squad':
            question = self.original_dataset[index]['question']
            original_answer = self.original_dataset[index]['context']

        if self.dataset_name == 'squad':
            if index % 2 == 0:
                answer = original_answer
                label = 0
            else:
                answer = self.answer(question)
                label = 1

            tokenized_properties = tokenize_function_artificial({'text': answer })

        else:
            if index % 2 == 0:
                text = original_text
                label = 0
            else:
                text = self.rewrite(original_text)
                label = 1

            tokenized_properties = tokenize_function_artificial({ 'text': text })
        if self.dataset_name == 'squad':
            result = {'question': question,
                      "answer": answer,
                      'label': label,
                      'pre_llm_text': original_answer}
        else:
            result = { 'text': text, 'label': label, 'pre_llm_text': original_text }
        result.update(tokenized_properties)

        return result
    
    def get_text_key(self):
        if get_args().source_dataset_type == "amazon_polarity":
            return "content"
        
        return "text"
    
    def rewrite(self, text: str):
        # if not text.startswith('"'):
        #     text = f'"{text}'
        # if not text.endswith('"'):
        #     text = f'{text}"'

        messages = [
            {"role": "user", "content": f"rewrite the following text: {text}"},
        ]
        result = self.pipe(messages, max_length=2000)
        rewritten_text = result[0]['generated_text'][1]['content']
        return rewritten_text

    def answer(self, text):
        messages = [
            {"role": "user", "content": f"answer this question: {text}"},
        ]
        result = self.pipe(messages, max_length=2000)
        rewritten_text = result[0]['generated_text'][1]['content']
        return rewritten_text

    def __len__(self):
        return len(self.original_dataset)