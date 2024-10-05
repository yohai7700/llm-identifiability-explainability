from peft import LoraConfig, TaskType, get_peft_model
from testing.model import model

lora_config = LoraConfig(
    task_type = TaskType.SEQ_CLS,
    r = 4,
    lora_alpha=32,
    lora_dropout=0.01,
    target_modules=['q_lin']
)

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()