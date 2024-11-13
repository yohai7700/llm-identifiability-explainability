from datasets import load_dataset
from torch.utils.data import Subset

from args import get_args

def load_text_datasets(dataset_name):
    if dataset_name == "imdb":
        dataset = load_dataset("shawhin/imdb-truncated", cache_dir=get_args().cache_dir)
        train_dataset, eval_dataset = dataset['train'], dataset['validation']
    elif dataset_name == "yelp":
        dataset = load_dataset("Yelp/yelp_review_full", cache_dir=get_args().cache_dir)
        train_dataset, eval_dataset = dataset['train'], dataset['test']
    elif  dataset_name == 'amazon_polarity':
        dataset = load_dataset("amazon_polarity", cache_dir=get_args().cache_dir)
        train_dataset, eval_dataset = dataset['train'], dataset['test']
    elif dataset_name == 'squad':
        dataset = load_dataset("squad", cache_dir=get_args().cache_dir)
        train_dataset, eval_dataset = dataset['train'], dataset['validation']

    train_dataset = Subset(train_dataset, range(get_args().training_subset_size))
    eval_dataset = Subset(eval_dataset, range(get_args().eval_subset_size))
    return train_dataset, eval_dataset