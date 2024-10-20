import torch

from transformers import pipeline
from model import tokenize_function_artificial

class ArtificialLlmTextDataset(torch.utils.data.Dataset):
    """Some Information about ArtificialLlmTextDataset"""
    def __init__(self, original_dataset: torch.utils.data.Dataset):
        super(ArtificialLlmTextDataset, self).__init__()

        self.original_dataset = original_dataset
        # Load model directly
        # tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)
        # model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)
        # self.llm = AutoModelForCausalLM.from_pretrained("gpt2")
        # self.llm_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # self.llm.generation_config.pad_token_id = self.llm_tokenizer.pad_token_id
        self.pipe = pipeline("text-generation", model="Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True, device_map="auto")

    def __getitem__(self, index):
        original_text = self.original_dataset[index]['text']
        
        if index % 2 == 0:
            text = original_text
            label = 0
        else:
            text = self.rewrite(original_text)
            label = 1

        tokenized_properties = tokenize_function_artificial({ 'text': text })
        result = { 'text': text, 'label': label }
        result.update(tokenized_properties)

        return result
    
    def rewrite(self, text):
        # prompt = f'rewrite: {text}'
        # tokenized_inputs = self.llm_tokenizer(
        #     prompt,
        #     truncation=True,
        #     max_length=512, 
        #     return_tensors="pt"
        # )
        # llm_tokens = self.llm.generate(
        #     tokenized_inputs['input_ids'],
        #     do_sample=True,
        #     temperature=0.9,
        #     max_length=1024,
        #     attention_mask=tokenized_inputs['attention_mask'] 
        # )
        # return self.llm_tokenizer.batch_decode(llm_tokens)[0][:200]
        messages = [
            {"role": "user", "content": f"rewrite the following text: {text}"},
        ]
        result = self.pipe(messages, max_length=2000)
        rewritten_text = result[0]['generated_text'][1]['content']
        return rewritten_text

    def __len__(self):
        return len(self.original_dataset)