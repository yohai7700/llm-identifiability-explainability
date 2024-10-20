from datasets import load_dataset
from torch.utils.data import Subset

from args import get_args

imdb_dataset = load_dataset("shawhin/imdb-truncated", cache_dir=get_args().cache_dir)
yelp_dataset = load_dataset("Yelp/yelp_review_full", cache_dir=get_args().cache_dir)

train_dataset = Subset(yelp_dataset['train'], range(1000))
eval_dataset = Subset(yelp_dataset['test'], range(200))