import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import argparse
from model import PointNet
from dataloader import DatasetWrapper
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main(args):
    with open('./dataset/ModelNet40/cls_dict.pkl', 'rb') as f:
        cls_dict = pickle.load(f)
    num_classes = len(cls_dict)

    model = PointNet(num_classes=num_classes)
    model.to(device)

    datasetwrapper = DatasetWrapper(batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    valid_ratio=args.valid_ratio)
    train_loader, valid_loader = datasetwrapper.get_train_valid_dataloaders()

    optimizer = optim.Adam(model.parameters(), 3e-4, weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader)
        for x, y in pbar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pbar = tqdm(valid_loader)
            for x, y in pbar:
                x, y = x.to(device), y.to(device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PointNet')
    parser.add_argument(
        '--epochs',
        type=int,
        default=200
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4
    )
    parser.add_argument(
        '--valid_ratio',
        type=float,
        default=0.1
    )
    args = parser.parse_args()

    main(args)