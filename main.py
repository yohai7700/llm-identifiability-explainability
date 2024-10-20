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

print_args()

if get_args().task == 'preprocess':
    from preprocessing import preprocess
    preprocess()
elif get_args().task == 'train':
    from training import trainer
    trainer.train()
elif get_args().task == 'test':
    from transformers import pipeline
    messages = [
        {"role": "user", "content": f"rewrite the following text:I love to play video games in the afternoon!"},
    ]
    pipe = pipeline("text-generation", model="Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True, device_map="auto")
    results = pipe(messages, max_length=1024)
    for result in results[0]['generated_text']:
        print(f"{result['role']}: {result['content']}")
else:
    print("Unsupported task currently. Please choose 'preprocess', 'test' or 'train'.")