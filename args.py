from dataclasses import dataclass
import argparse
import torch

@dataclass
class Args:
    data_path: str
    batch_size: int
    epochs: int
    lr: float
    optimizer: str
    seed: int
    log_dir: str
    task: str
    cache_dir: str
    cache_user: str
    classification_model_name: str
    llm_generating_model_name: str
    training_subset_size: int
    eval_subset_size: int
    lora_rank: int
    device: torch.device
    source_dataset_type: str
    training_dataset_type: str
    eval_dataset_type: str


__KNOWN_CACHE_DIRS = {
    "yohai": "/home/sharifm/teaching/tml-0368-4075/2024-spring/students/yohaimazuz/.cache/torch/hub"
}

args: Args = None

def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="LLM Detection - Arguments")

    dataset_types = ["yelp", "imdb", "amazon_polarity"]

    # Environment Arguments
    parser.add_argument("--device", type=str, default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), help="Random seed for reproducibility")

    # Task Arguments
    parser.add_argument("--task",
        type=str, 
        required=True,
        help="Task to perform: train, test, predict, preprocess, evaluate, interpret or persist_to_csv", 
        choices=["train", "test", "predict", "preprocess", "persist_to_csv", "evaluate", "interpret"]
    )
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="./data", help="Path to the dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--cache_user", type=str, default=None, help="Use the cache directory of this user", choices=["yohai"])
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for loading datasets")
    parser.add_argument("--training_subset_size", type=int, default=5000, help="Size of the training subset")
    parser.add_argument("--eval_subset_size", type=int, default=1000, help="Size of the eval subset")
    parser.add_argument("--source_dataset_type", type=str, default="yelp", help="Source dataset type", choices=dataset_types)
    
    # Model arguments
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    parser.add_argument("--weights_folder_path", action="store_true", help="Path for storing weights")
    parser.add_argument("--llm_generating_model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct", help="Model name for LLM generation")
    parser.add_argument("--classification_model_name", type=str, default="distilbert-base-uncased", help="Model name for LLM generation")

    # Lora arguments
    parser.add_argument("--lora_rank", default=8, action="store_true", help="Rank used for LoRA algorithm")

    # Prediction arguments
    parser.add_argument("--prediction_text_input", type=str, default=None, help="Text input for prediction")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer to use")
    parser.add_argument("--training_dataset_type", type=str, default="yelp", help="Training dataset type", choices=dataset_types)
    parser.add_argument("--eval_dataset_type", type=str, default="yelp", help="Eval dataset type", choices=dataset_types)
    
    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save logs")
    parser.add_argument("--dataset_name", type=str, default="squad", help="dataset_name")
    parsed_args = parser.parse_args()

    if parsed_args.cache_dir is None and parsed_args.cache_user is not None:
        parsed_args.cache_dir = __KNOWN_CACHE_DIRS.get(parsed_args.cache_user, None)

    return parsed_args

def get_args():
    global args
    if args is None:
        args = parse_args()
    return args

def print_args():
    global args
    print("Project Configuration:")
    if args is None:
        args = get_args()
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")