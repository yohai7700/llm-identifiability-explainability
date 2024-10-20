from datasets import load_dataset
from torch.utils.data import Subset

from args import get_args

YOHAI_CACHE_DIR = "/home/sharifm/teaching/tml-0368-4075/2024-spring/students/yohaimazuz/.cache"
cache_dir = YOHAI_CACHE_DIR if get_args().cache_user == "yohai" else None

print(f"Using cache directory: {cache_dir}")

imdb_dataset = load_dataset("shawhin/imdb-truncated", cache_dir=cache_dir)
yelp_dataset = load_dataset("Yelp/yelp_review_full", cache_dir=cache_dir)

train_dataset = Subset(yelp_dataset['train'], range(1000))
eval_dataset = Subset(yelp_dataset['test'], range(200))