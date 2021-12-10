from torch.utils.data import Dataset
import os.path as osp
import torchaudio
from utils import LABELS_10, LABELS_3, LABELS_10_NEW
import torch
import librosa
import random


class BasicDataset(Dataset):
    def __init__(self, data_dir, features_dir, train_df, n_classes, test, augmentations):
        self.data_dir = data_dir
        self.features_dir = features_dir
        self.file_names = train_df['filename']
        self.labels = train_df['scene_label']
        self.n_classes = n_classes
        self.test = test
        self.augmentations = augmentations

    def __len__(self):
        return self.file_names.shape[0]

    def __getitem__(self, item):
        audio_file = osp.basename(self.file_names.iloc[item])
        assert osp.exists(osp.join(self.data_dir, audio_file)) == 1, f"audio file {osp.join(self.data_dir, audio_file)} doesn't exist "
        if self.test:
            aug_num = 0
        elif self.n_classes == 10:
            aug_num = random.randint(0, 3)

        mel = torch.load(osp.join(self.features_dir, f'{audio_file.replace(".wav", "")}_mel_{aug_num}.pkl').replace('\\','/'))

        if self.n_classes == 10:
            label = LABELS_10[self.labels.iloc[item]]
        return {
            'mels': mel.type(torch.FloatTensor),
            'label': torch.tensor(label)
        }