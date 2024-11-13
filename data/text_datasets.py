from datasets import load_dataset
from torch.utils.data import Subset

from args import get_args

def load_text_datasets(dataset_type = get_args().source_dataset_type):
    if dataset_type == "imdb":
        dataset = load_dataset("shawhin/imdb-truncated", cache_dir=get_args().cache_dir)
        train_dataset, eval_dataset = dataset['train'], dataset['validation']
    elif dataset_type == "yelp":
        dataset = load_dataset("Yelp/yelp_review_full", cache_dir=get_args().cache_dir)
        train_dataset, eval_dataset = dataset['train'], dataset['test']
    elif dataset_type == "amazon_polarity":
        dataset = load_dataset("amazon_polarity", cache_dir=get_args().cache_dir)
        train_dataset, eval_dataset = dataset['train'], dataset['test']
        
    train_dataset = Subset(train_dataset, range(get_args().training_subset_size))
    eval_dataset = Subset(eval_dataset, range(get_args().eval_subset_size))
    return train_dataset, eval_dataset