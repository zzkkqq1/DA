import numpy as np
import torch
from torch.utils.data import Dataset


def make_windows(disp, rain, seq_len=24, horizon=1):
    """
    Construct sliding-window samples.

    disp: [T, N, 2], displacement sequence of N monitoring points
    rain: [T, 1], rainfall sequence

    return:
        x_disp: [S, seq_len, N, 2]
        x_rain: [S, seq_len, 1]
        y_x:    [S, N]
        y_y:    [S, N]
    """
    xs_disp, xs_rain, ys_x, ys_y = [], [], [], []

    max_start = len(disp) - seq_len - horizon + 1

    for i in range(max_start):
        xs_disp.append(disp[i:i + seq_len])
        xs_rain.append(rain[i:i + seq_len])

        target_t = i + seq_len + horizon - 1
        ys_x.append(disp[target_t, :, 0])
        ys_y.append(disp[target_t, :, 1])

    return (
        np.asarray(xs_disp, dtype=np.float32),
        np.asarray(xs_rain, dtype=np.float32),
        np.asarray(ys_x, dtype=np.float32),
        np.asarray(ys_y, dtype=np.float32),
    )


class SlopeDataset(Dataset):
    def __init__(self, x_disp, x_rain, y_x, y_y):
        self.x_disp = torch.tensor(x_disp, dtype=torch.float32)
        self.x_rain = torch.tensor(x_rain, dtype=torch.float32)
        self.y_x = torch.tensor(y_x, dtype=torch.float32)
        self.y_y = torch.tensor(y_y, dtype=torch.float32)

    def __len__(self):
        return len(self.y_x)

    def __getitem__(self, idx):
        return self.x_disp[idx], self.x_rain[idx], self.y_x[idx], self.y_y[idx]