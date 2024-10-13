from datasets import load_dataset, DatasetDict

from args import get_args

YOHAI_CACHE_DIR = "/home/sharifm/teaching/tml-0368-4075/2024-spring/students/yohaimazuz/.cache"
cache_dir = YOHAI_CACHE_DIR if get_args().cache_user == "yohai" else None

imdb_dataset = load_dataset("shawhin/imdb-truncated", cache_dir=cache_dir)
yelp_dataset = load_dataset("Yelp/yelp_review_full", cache_dir=cache_dir)

dataset = ds = DatasetDict({
    'train': yelp_dataset['train'],
    'validation': yelp_dataset['test']
})