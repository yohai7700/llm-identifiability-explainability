import torch
import os
from tqdm import tqdm

from args import get_args
from data.artificial_llm_text_dataset import ArtificialLlmTextDataset
from data.utils.dataset_csv_utils import save_dataset_to_csv
from data.text_datasets import load_text_datasets
from data.list_dataset import ListDataset

def get_model_alias(model_name: str):
    if model_name == "Qwen/Qwen2-0.5B-Instruct":
        return "qwen2-0.5b-instruct"
    if model_name == "microsoft/Phi-3.5-mini-instruct":
        return "phi-3.5-mini-instruct"
    if model_name == "meta-llama/Llama-3.2-1B-Instruct":
        return "llama-3.2-1b-instruct"
    
    raise ValueError(f"Model '{model_name}' is not supported")

def get_preprocessed_dataset_folder_path():
    dataset_type = get_args().source_dataset_type
    model = get_model_alias(get_args().llm_generating_model_name)
    return f'./data/checkpoints/{dataset_type}_{model}'

def get_preprocessed_dataset_path(label: str):
    folder_path = get_preprocessed_dataset_folder_path()
    return f'{folder_path}/{label}_dataset.pt'

def preprocess():
    folder_path = get_preprocessed_dataset_folder_path()
    dataset_type = get_args().source_dataset_type
    os.makedirs(folder_path, exist_ok=True)
    train_dataset, eval_dataset = load_text_datasets(dataset_type)
    for label, dataset in [('train', train_dataset), ('eval', eval_dataset)]:
        # if label == 'train':
        #     continue
        llm_dataset = ArtificialLlmTextDataset(dataset, dataset_type)

        print(f"Preprocessing {label} dataset...")
        data_items = [llm_dataset[i] for i in tqdm(range(len(llm_dataset)))]
        
        torch.save(data_items, f'{folder_path}/{label}_dataset.pt')

    persist_to_csv()

def persist_to_csv():
    dataset_type = get_args().source_dataset_type
    train_dataset, eval_dataset = load_text_datasets(dataset_type)
    os.makedirs(f'{get_preprocessed_dataset_folder_path()}/csv', exist_ok=True)
    for label, dataset in [('train', train_dataset), ('eval', eval_dataset)]:
        items = torch.load(get_preprocessed_dataset_path(label))
        dataset = ListDataset(items)

        save_dataset_to_csv(dataset, f'{get_preprocessed_dataset_folder_path()}/csv/{label}_dataset.csv')
