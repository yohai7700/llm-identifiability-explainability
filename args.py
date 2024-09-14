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

args: Args = None

def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="PaceMaker Prediction - Arguments")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, default="./data", help="Path to the dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    
    # Model arguments
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")
    parser.add_argument("--weights_folder_path", action="store_true", help="Path for storing weights")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer to use")
    
    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory to save logs")

    return parser.parse_args()

def get_args():
    global args
    if args is None:
        args = parse_args()
    return args

def print_args():
    global args
    print("Project Configuration:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")