import argparse
import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from utils import drc
from librosa.feature import melspectrogram, delta
from librosa.effects import pitch_shift
import soundfile as sf


# create 3D log_melspectrogram
def create_mels_deltas(waveform, sample_rate):
    one_mel = melspectrogram(waveform.squeeze(0).numpy(), sr=sample_rate, n_fft=2048, hop_length=1024,
                             n_mels=128, fmin=0.0, fmax=sample_rate / 2, htk=True, norm=None)
    one_mel = np.log(one_mel + 1e-8)
    one_mel = (one_mel - np.min(one_mel)) / (np.max(one_mel) - np.min(one_mel))
    one_mel_delta = delta(one_mel)
    one_mel_delta = (one_mel_delta - np.min(one_mel_delta)) / (np.max(one_mel_delta) - np.min(one_mel_delta))
    one_mel_delta_delta = delta(one_mel, order=2)
    one_mel_delta_delta = (one_mel_delta_delta - np.min(one_mel_delta_delta)) / (
            np.max(one_mel_delta_delta) - np.min(one_mel_delta_delta))
    mel_3d = torch.cat([torch.tensor(one_mel).unsqueeze(0), torch.tensor(one_mel_delta).unsqueeze(0),
                        torch.tensor(one_mel_delta_delta).unsqueeze(0)], dim=0)
    return mel_3d


def create_features(audio_path, aug_index):
    # waveform.shape = [channel, time]
    waveform, sample_rate = torchaudio.load(audio_path)
    if aug_index == 1:
        waveform = waveform + torch.randn(waveform.shape) * 0.001
    if aug_index == 2:
        waveform = torchaudio.functional.contrast(waveform)
    if aug_index == 3:
        if waveform.shape[0] == 2:
            waveform = torch.cat([torch.tensor(pitch_shift(waveform.squeeze(0).numpy()[0, :],
                                                           sample_rate, np.random.random(1)[0])).unsqueeze(0),
                                  torch.tensor(pitch_shift(waveform.squeeze(0).numpy()[1, :],
                                                           sample_rate, np.random.random(1)[0])).unsqueeze(0)], dim=0)
        else:
            waveform = torch.tensor(pitch_shift(waveform.squeeze(0).numpy(),
                                                sample_rate, np.random.random(1)[0])).unsqueeze(0)
    if aug_index == 4:
        waveform = drc(waveform, bitdepth=6)
    if waveform.shape[0] == 2:
        full_mel_3d = torch.cat(
            [create_mels_deltas(waveform[0], sample_rate), create_mels_deltas(waveform[1], sample_rate)], dim=0)
    else:
        full_mel_3d = create_mels_deltas(waveform, sample_rate)
    return full_mel_3d


if __name__ == '__main__':
    print("test!")
    parser = argparse.ArgumentParser(description='3 or 10')
    parser.add_argument('-n', '--n_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--data_path', '--data_path', type=str, default='./data', help='Path to data direction')
    args = parser.parse_args()
    if args.n_classes == 10:
        data_dir = osp.join(args.data_path, 'TAU-urban-acoustic-scenes-2020-mobile-development/')
    feature_dir = osp.join(args.data_path, 'mel_features_3d')
    if not osp.exists(feature_dir):
        os.mkdir(feature_dir)
    if args.n_classes == 10:
        meta_csv_file = './data/TAU-urban-acoustic-scenes-2020-mobile-development/meta.csv'
    meta_csv_df = pd.read_csv(meta_csv_file, sep='\t')
    aug_num = 4
    for audio_file in tqdm(meta_csv_df['filename']):
        for i in range(aug_num):
            feature_pkl = osp.join(feature_dir, f'{osp.basename(audio_file).replace(".wav", "")}_mel_{i}.pkl')
            if osp.exists(feature_pkl):
                continue
            audio_path = osp.join(data_dir, audio_file)
            mel_3d = create_features(audio_path, i)
            torch.save(mel_3d, feature_pkl)