import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from testing.model import tokenize_function, tokenize_function_artificial

class ArtificialLlmTextDataset(torch.utils.data.Dataset):
    """Some Information about ArtificialLlmTextDataset"""
    def __init__(self, original_dataset: torch.utils.data.Dataset):
        super(ArtificialLlmTextDataset, self).__init__()

        self.original_dataset = original_dataset
        self.llm = AutoModelForCausalLM.from_pretrained("gpt2")
        self.llm_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def __getitem__(self, index):
        original_text = self.original_dataset[index]['text']
        
        if index % 2 == 0:
            text = original_text
            label = 0
        else:
            tokenized_inputs = self.llm_tokenizer(f'rewrite {original_text}', return_tensors="pt").input_ids
            llm_tokens = self.llm.generate(
                tokenized_inputs,
                do_sample=True,
                temperature=0.9,
                max_new_tokens=200
            )
            text = self.llm_tokenizer.batch_decode(llm_tokens)[0]
            label = 1

        tokenized_properties = tokenize_function_artificial({ 'text': text })
        result = { 'text': text, 'label': label }
        result.update(tokenized_properties)

        print(f'success: {index}')
        return result

    def __len__(self):
        return len(self.original_dataset)