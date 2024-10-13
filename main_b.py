import torch

import transformers
import accelerate
import peft
from args import print_args, get_args

print(f"Transformers version: {transformers.__version__}")
print(f"Accelerate version: {accelerate.__version__}")
print(f"PEFT version: {peft.__version__}")
print(f"CUDA is available: {torch.cuda.is_available()}")
print(f'Cache User: {get_args().cache_user}')

from testing.training import trainer

print_args()

trainer.train()

# from transformers import AutoModelForCausalLM, AutoTokenizer

# model = AutoModelForCausalLM.from_pretrained("gpt2-medium", torch_dtype=torch.float16)
# tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")

# prompt = 'rewrite the following text "this food is very tasty"'

# model_inputs = tokenizer([prompt], return_tensors="pt")

# generated_ids = model.generate(**model_inputs, max_new_tokens=20, do_sample=True)
# print(tokenizer.batch_decode(generated_ids)[0])