import torch
from tqdm import tqdm

from artificial_llm_text_dataset import ArtificialLlmTextDataset
from csv_dataset import save_dataset_to_csv
from dataset import train_dataset, eval_dataset
from list_dataset import ListDataset

def preprocess():
    llm_train_dataset = ArtificialLlmTextDataset(train_dataset)
    llm_eval_dataset = ArtificialLlmTextDataset(eval_dataset)

    print("Preprocessing train dataset...")
    train_items = [llm_train_dataset[i] for i in tqdm(range(len(llm_train_dataset)))]

    print("Preprocessing eval dataset...")
    eval_items = [llm_train_dataset[i] for i in tqdm(range(len(llm_eval_dataset)))]
    
    torch.save(train_items, './dataset_checkpoints/yelp/train_dataset.pt')
    torch.save(eval_items, './dataset_checkpoints/yelp/eval_dataset.pt')

def persist_to_csv():
    train_items = torch.load('./dataset_checkpoints/yelp/train_dataset.pt')
    dataset = ListDataset(train_items)

    save_dataset_to_csv(dataset, './dataset_checkpoints/yelp/train_dataset.csv')