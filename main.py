import torch

from args import print_args, get_args

import os

print_args()

if get_args().cache_dir is not None:
    torch.hub.set_dir(get_args().cache_dir)
    os.environ['HF_HOME'] = get_args().cache_dir
    
import transformers
import accelerate
import peft

print(f"Transformers version: {transformers.__version__}")
print(f"Accelerate version: {accelerate.__version__}")
print(f"PEFT version: {peft.__version__}")
print(f"CUDA is available: {torch.cuda.is_available()}")
print(f'Cache User: {get_args().cache_user}')

if get_args().task == 'preprocess':
    from data.utils.preprocessing import preprocess
    preprocess()
elif get_args().task == 'persist_to_csv':
    from data.utils.preprocessing import persist_to_csv
    persist_to_csv()
elif get_args().task == 'train':
    from training.trainer import trainer
    trainer.train()
    trainer.save_model('./models/checkpoints/llm_cls/distilbert_qwen/model')
elif get_args().task == 'predict':
    from prediction import predict
    print(predict())
elif get_args().task == 'interpret':
    from interpretation import interpret
    interpret()
elif get_args().task == 'evaluate':
    from training.evaluation import eval
    eval()
elif get_args().task == 'test':
    from testing import test
    test()
else:
    print(f"Unsupported task: {get_args().task}")
