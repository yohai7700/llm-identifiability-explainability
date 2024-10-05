import torch

import transformers
import accelerate
import peft

print(f"Transformers version: {transformers.__version__}")
print(f"Accelerate version: {accelerate.__version__}")
print(f"PEFT version: {peft.__version__}")
print(f"CUDA is available: {torch.cuda.is_available()}")

from testing.training import trainer

trainer.train()