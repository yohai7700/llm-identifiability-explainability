import torch
from torch import nn, optim
import torch.utils.data

from tqdm import tqdm

from args import Args

def train_model(args: Args, model: nn.Module, dataloader: torch.utils.data.DataLoader):
    optimizer = optim.adam.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
        
    for i in tqdm(range(args.epochs)):
        for x, y in dataloader:
            optimizer.zero_grad()
            
            prediction = model()
            loss = criterion(prediction, y)
            loss.backward()
            
            optimizer.step()
            
            if 