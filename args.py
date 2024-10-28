from dataclasses import dataclass
import argparse

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
    llm_generating_model_name: str
    training_subset_size: int
    eval_subset_size: int


__KNOWN_CACHE_DIRS = {
    "yohai": "/home/sharifm/teaching/tml-0368-4075/2024-spring/students/yohaimazuz/.cache/torch/hub"
}

args: Args = None

def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="LLM Detection - Arguments")

    # Task Arguments
    parser.add_argument("--task",
        type=str, 
        required=True,
        help="Task to perform: train, test, predict, preprocess or persist_to_csv", 
        choices=["train", "test", "predict", "preprocess", "persist_to_csv"]
    )
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="./data", help="Path to the dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--cache_user", type=str, default=None, help="Use the cache directory of this user", choices=["yohai"])
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for loading datasets")
    parser.add_argument("--training_subset_size", type=int, default=2000, help="Size of the training subset")
    parser.add_argument("--eval_subset_size", type=int, default=200, help="Size of the eval subset")
    
    # Model arguments
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    parser.add_argument("--weights_folder_path", action="store_true", help="Path for storing weights")
    parser.add_argument("--llm_generating_model_name", default="Qwen/Qwen2-0.5B-Instruct", action="store_true", help="Model name for LLM generation")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer to use")
    
    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save logs")

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