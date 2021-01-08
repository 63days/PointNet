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

    scheduler = optim.lr_scheduler.StepLR(model.optimizer, step_size=20, gamma=0.5)

    train_losses = []
    val_losses = []
    acc_list = []
    best_acc = -1

    for epoch in range(args.epochs):
        train_loss = []
        val_loss = []
        acc = []
        model.train()
        pbar = tqdm(train_loader)
        for x, y in pbar:
            loss = model.train_batch(x, y)
            train_loss.append(loss)
            pbar.set_description(f'E:{epoch+1:3d}|L:{loss:.4f}|lr:{scheduler.get_last_lr()[0]:.2e}')

        train_loss = sum(train_loss) / len(train_loss)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            for x, y in valid_loader:
                acc_batch, loss = model.valid_batch(x, y)
                acc.append(acc_batch)
                val_loss.append(loss)

            val_loss = sum(val_loss) / len(val_loss)
            val_losses.append(val_loss)

            acc = sum(acc) / len(acc)
            print(f'Acc:{acc*100:.2f} VL:{val_loss:.4f}')
            acc_list.append(acc)

        scheduler.step()

        if best_acc < acc:
            best_acc = acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'best_acc': best_acc
            }, './checkpoint/pointnet.ckpt')

        if (epoch+1) % 10 == 0:
            torch.save({
                'train_losses': train_losses,
                'val_losses': val_losses,
                'acc_list': acc_list,
                'best_acc': best_acc
            }, 'results.ckpt')

    torch.save({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'acc_list': acc_list,
        'best_acc': best_acc
    }, 'results.ckpt')


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
        default=32
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