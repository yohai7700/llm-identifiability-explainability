import torch
from torch.utils.data import Subset

from datasets import load_dataset

from args import get_args

def load_text_datasets():
    dataset_type = 'yelp'
    if dataset_type == "imdb":
        dataset = load_dataset("shawhin/imdb-truncated", cache_dir=get_args().cache_dir)
        train_dataset, eval_dataset = dataset['train'], dataset['validation']
    elif dataset_type == "yelp":
        dataset = load_dataset("Yelp/yelp_review_full", cache_dir=get_args().cache_dir)
        train_dataset, eval_dataset = dataset['train'], dataset['test']
    
    training_indices = torch.randperm(len(train_dataset)).tolist()[:get_args().training_subset_size]
    eval_indices = torch.randperm(len(eval_dataset)).tolist()[:get_args().eval_subset_size]
    
    train_dataset = Subset(train_dataset, training_indices)
    eval_dataset = Subset(eval_dataset, eval_indices)
    return train_dataset, eval_dataset