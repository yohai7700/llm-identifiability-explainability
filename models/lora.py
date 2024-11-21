from peft import LoraConfig, TaskType, get_peft_model

from args import get_args

def get_target_modules():
    model_name = get_args().classification_model_name
    if model_name == 'bert-base-uncased':
        return ['query']
    return ['q_lin', "k_lin", "v_lin"]

def attach_lora(model, tokenizer):
    lora_config = LoraConfig(
        task_type = TaskType.SEQ_CLS,
        r = get_args().lora_rank,
        lora_alpha=32,
        lora_dropout=0.01,
        target_modules=get_target_modules()
    )

    lora_model = get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()

    lora_model.resize_token_embeddings(len(tokenizer))
    
    return lora_model