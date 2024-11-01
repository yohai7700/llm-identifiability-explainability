import torch
import os
from tqdm import tqdm

from data.artificial_llm_text_dataset import ArtificialLlmTextDataset
from data.utils.dataset_csv_utils import save_dataset_to_csv
from data.text_datasets import load_text_datasets
from data.list_dataset import ListDataset

def preprocess():
    os.makedirs('./data/checkpoints/yelp', exist_ok=True)
    train_dataset, eval_dataset = load_text_datasets()
    for label, dataset in [('train', train_dataset), ('eval', eval_dataset)]:
        llm_dataset = ArtificialLlmTextDataset(dataset)

        print(f"Preprocessing {label} dataset...")
        data_items = [llm_dataset[i] for i in tqdm(range(len(llm_dataset)))]
        
        torch.save(data_items, f'./data/checkpoints/yelp/{label}_dataset.pt')

def persist_to_csv():
    train_dataset, eval_dataset = load_text_datasets()
    for label, dataset in [('train', train_dataset), ('eval', eval_dataset)]:
        items = torch.load(f'./data/checkpoints/yelp/{label}_dataset.pt')
        dataset = ListDataset(items)

        save_dataset_to_csv(dataset, f'./data/checkpoints/yelp/{label}_dataset.csv')