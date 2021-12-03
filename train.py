from utils import plot_loss_score, mixup
import pickle

import pandas as pd
import os.path as osp
from dataset import BasicDataset
import torch
from torch.utils.data import DataLoader
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from eval import eval_net
from focal_loss import FocalLoss
from sklearn.metrics import accuracy_score
import random
from utils import label_10_to_3, vec3_to_vec10


def train_net(net, epochs, data_dir, output_dir, features_dir, folds_dir, dir_checkpoint, batch_size, lr, device,
              n_classes,
              timestamp, setup=None, augmentations='all'):
    # load folds
    train_csv = osp.join(folds_dir, 'fold1_train.csv')
    val_csv = osp.join(folds_dir, 'fold1_evaluate.csv')
    train_df = pd.read_csv(train_csv, sep='\t')
    val_df = pd.read_csv(val_csv, sep='\t')

    dataset_train = BasicDataset(data_dir, features_dir, train_df, n_classes, test=False, augmentations=augmentations)
    dataset_val = BasicDataset(data_dir, features_dir, val_df, n_classes, test=True, augmentations=augmentations)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                            drop_last=True)

    print(f'''\nStarting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {round(train_df.shape[0] / (train_df.shape[0] + val_df.shape[0]) * 100)}%
            Validation size: {round(val_df.shape[0] / (train_df.shape[0] + val_df.shape[0]) * 100)}%
            Checkpoints dir: {dir_checkpoint}
            Device:          {device.type}
        ''')

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = FocalLoss()

    loss_val = []
    loss_train = []
    score_val = []
    score_train = []

    for epoch in range(epochs):
        net.train()
        epoch_loss = []
        all_score_train = []
        with tqdm(total=train_df.shape[0], desc=f'Epoch{epoch + 1} / {epochs}', unit='img') as pbar:
            for index, batch in enumerate(train_loader):
                mels = batch['mels']
                label = batch['label']
                mels = mels.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.long)
                _label = label
                if setup == 'two_path':
                    pred_vec10, pred_vec3 = net(mels)
                    loss = criterion(pred_vec10 * vec3_to_vec10(pred_vec3, device), label)
                    test = criterion()
                else:
                    pred_vec = net(mels)
                    loss = criterion(pred_vec, label)

                epoch_loss.append(loss)
                pbar.set_postfix(**{'loss (batch) ': loss})
                optimizer.zero_grad()
                loss.backward()

                if np.mod(index, 100) != 0:
                    # set training to False
                    net.eval()
                    if setup == 'two_path':
                        pred_vec10_eval, pred_vec3_eval = net(mels)
                        all_score_train.append(accuracy_score(
                            (pred_vec10_eval * vec3_to_vec10(pred_vec3_eval, device)).argmax(dim=1).cpu().numpy(),
                            _label.flatten().cpu().numpy()))
                    else:
                        pred_vec_eval = net(mels)
                        all_score_train.append(
                            accuracy_score(pred_vec_eval.argmax(dim=1).flatten().cpu().numpy(),
                                           _label.flatten().cpu().numpy()))

                    net.train()
                optimizer.step()
                pbar.update(mels.shape[0])

            val_loss, val_score = eval_net(net, val_loader, device, criterion, n_classes, setup)
            score_val.append(val_score)
            loss_val.append(val_loss.item())
            score_train.append(sum(all_score_train) / len(all_score_train))
            loss_train.append(sum(epoch_loss) / len(epoch_loss))
            torch.save(net.state_dict(),
                       osp.join(dir_checkpoint, f'CP_epoch{epoch + 1}.pth'))
            print(f'epoch = {epoch + 1}, train loss = {sum(epoch_loss) / len(epoch_loss)},'
                  f' validation loss = {val_loss}',
                  file=open(osp.join(output_dir, f'log_{timestamp}.txt'), 'a'))
            print(f'saved weights to {osp.join(dir_checkpoint, f"CP_epoch_{epoch + 1}.pth")}',
                  file=open(osp.join(output_dir, f'log_{timestamp}.txt'), 'a'))

            # plot_loss_score(np.arange(1, epoch + 2).astype(int), loss_train, loss_val, timestamp, 'loss', output_dir)
            # plot_loss_score(np.arange(1, epoch + 2).astype(int), score_train, score_val, timestamp, 'score', output_dir)

            with open(osp.join(output_dir, f'score_train_{timestamp}.pkl'), 'wb') as handle:
                pickle.dump(score_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(osp.join(output_dir, f'score_val_{timestamp}.pkl'), 'wb') as handle:
                pickle.dump(score_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
