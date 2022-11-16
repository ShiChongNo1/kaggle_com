import logging
import math
import random

import evaluate
import torch
from easydict import EasyDict
from torch import nn
from torch.utils.data import DataLoader
from models.LSTM import LstmRNN
from models.TransAm import TransAm
from models.model import Transformer
from pretreatment.datamachine import scale
from utils.plot import *
from utils.helpers import *
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
                    datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)


def flip_from_probability(p):
    return True if random.random() < p else False


def train_epoch(config: EasyDict,
                model: nn.Module,
                dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                device,
                loss_function,
                step: int):
    train_loss = 0
    model.train()
    train_bar = tqdm(dataloader)
    train_bar.set_description(
        f'Epoch [{step + 1}/{config.train.epoch}] Training')
    optimizer.param_groups[0]['lr'] = 0.0001
    for _input, target in train_bar:
        sampled_src = _input.float().to(device)
        sampled_target = target.float().to(device)
        prediction = model(sampled_src)
        loss = loss_function(sampled_target, prediction)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.detach().item()
        train_bar.set_postfix({'loss': f'{loss:1.5f}'})
    return train_loss


def val_epoch(config: EasyDict,
              model: nn.Module,
              dataloader: DataLoader,
              device,
              step: int):
    model.eval()
    metrics = evaluate.load("mse")
    val_bar = tqdm(dataloader)
    acc = float(0)
    val_bar.set_description(
        f'Epoch [{step + 1}/{config.train.epoch}] Validation')
    for _input, target in val_bar:
        with torch.no_grad():
            prediction = model(_input.float().to(device))
            for i in range(len(prediction)):
                p = prediction[i].reshape([config.train.training_length])
                t = target.float().to(device)[i].reshape(
                    [config.train.training_length])
                metrics.add_batch(predictions=p, references=t)
            acc = metrics.compute()['mse']
            val_bar.set_postfix({'metrics': f"{100 * acc:.5f}%"})
    return acc


def train(config: EasyDict,
          dataloader,
          val_dataloader,
          path_to_save_model,
          path_to_save_loss,
          path_to_save_predictions,
          device):
    device = torch.device(device)
    epoch = config.train.epoch
    # model = LstmRNN(input_size=5, hidden_size=20, output_size=1, num_layers=10).double().to(device)
    # model = Transformer(feature_size=tl * 5)
    model = TransAm(feature_size=771).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200)
    best_model = ""
    min_train_loss = float('inf')
    min_acc = float('inf')

    for step in range(epoch + 1):
        train_loss = train_epoch(config=config,
                                 model=model,
                                 dataloader=dataloader,
                                 optimizer=optimizer,
                                 device=device,
                                 loss_function=criterion,
                                 step=step)
        # 将train_loss 写进日志
        log_loss(train_loss, path_to_save_loss, train=True)

        metrics = val_epoch(config=config,
                            model=model,
                            dataloader=val_dataloader,
                            device=device,
                            step=step)
        # 将验证集的 metrics 写进日志
        log_loss(metrics, path_to_save_loss, train=False)

        # train_loss < min_train_loss
        if metrics < min_acc:
            torch.save(model.state_dict(), path_to_save_model +
                       f"best_train_{epoch}.pth")
            torch.save(optimizer.state_dict(),
                       path_to_save_model + f"optimizer_{epoch}.pth")
            min_acc = metrics
            best_model = f"best_train_{epoch}.pth"

        if epoch % 5 == 0:
            for p in optimizer.param_groups:  # 更新每个group里的参数lr
                p['lr'] *= 0.9

    plot_loss(path_to_save_loss, train=True)
    return best_model
