import torch
import numpy as np


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x_disp, x_rain, y_x, y_y in loader:
        x_disp = x_disp.to(device)
        x_rain = x_rain.to(device)
        y_x = y_x.to(device)
        y_y = y_y.to(device)

        optimizer.zero_grad()

        pred_x, pred_y, _ = model(x_disp, x_rain)

        loss_x = criterion(pred_x, y_x)
        loss_y = criterion(pred_y, y_y)
        loss = loss_x + loss_y

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * len(y_x)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    for x_disp, x_rain, y_x, y_y in loader:
        x_disp = x_disp.to(device)
        x_rain = x_rain.to(device)
        y_x = y_x.to(device)
        y_y = y_y.to(device)

        pred_x, pred_y, _ = model(x_disp, x_rain)

        loss = criterion(pred_x, y_x) + criterion(pred_y, y_y)
        total_loss += loss.item() * len(y_x)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    pred_x_list, pred_y_list = [], []
    true_x_list, true_y_list = [], []
    alpha_list = []

    for x_disp, x_rain, y_x, y_y in loader:
        x_disp = x_disp.to(device)
        x_rain = x_rain.to(device)

        pred_x, pred_y, alpha = model(x_disp, x_rain)

        pred_x_list.append(pred_x.cpu().numpy())
        pred_y_list.append(pred_y.cpu().numpy())
        true_x_list.append(y_x.numpy())
        true_y_list.append(y_y.numpy())
        alpha_list.append(alpha.cpu().numpy())

    return (
        np.concatenate(pred_x_list, axis=0),
        np.concatenate(pred_y_list, axis=0),
        np.concatenate(true_x_list, axis=0),
        np.concatenate(true_y_list, axis=0),
        np.concatenate(alpha_list, axis=0),
    )