import torch
import os
from tqdm import tqdm

from args import get_args
from data.artificial_llm_text_dataset import ArtificialLlmTextDataset
from data.utils.dataset_csv_utils import save_dataset_to_csv
from data.text_datasets import load_text_datasets
from data.list_dataset import ListDataset

def preprocess(dataset_name):
    dataset_name = get_args().dataset_name
    os.makedirs('./data/checkpoints/squad', exist_ok=True)
    train_dataset, eval_dataset = load_text_datasets(dataset_name)
    for label, dataset in [('train', train_dataset), ('eval', eval_dataset)]:
        # if label == 'train':
        #     continue
        llm_dataset = ArtificialLlmTextDataset(dataset, dataset_name)

        print(f"Preprocessing {label} dataset...")
        data_items = [llm_dataset[i] for i in tqdm(range(len(llm_dataset)))]
        
        torch.save(data_items, f'./data/checkpoints/{dataset_name}/{label}_dataset.pt')

def persist_to_csv():
    dataset_name = get_args().dataset_name
    train_dataset, eval_dataset = load_text_datasets(dataset_name)
    for label, dataset in [('train', train_dataset), ('eval', eval_dataset)]:
        # if label == 'train':
        #     continue
        items = torch.load(f'./data/checkpoints/{dataset_name}/{label}_dataset.pt')
        dataset = ListDataset(items)

        save_dataset_to_csv(dataset, f'./data/checkpoints/{dataset_name}/{label}_dataset.csv')