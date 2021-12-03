import pickle
import torch


def test1():
    data = torch.load('./data/mel_features_3d/airport-barcelona-0-0-a_mel_0.pkl')
    print(data)
    print(data.shape)

test1()