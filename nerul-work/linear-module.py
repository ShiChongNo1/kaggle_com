import pandas as pd
import numpy as np
import torch
import evaluate
from src.loader import loader_database
from src.utils import save_model, load_model, Logger, same_seeds
from model.linear import linearModule
from torch.nn import BCELoss
from torch.optim import SGD, lr_scheduler
from accelerate import Accelerator
from tqdm.auto import tqdm
import matplotlib.pylab as plt
from torch.utils.tensorboard import SummaryWriter
from pylab import *


def train_model(train_loader, val_loader, input_num, writer):

    accelerator = Accelerator()
    accelerator.print(f'加载模型')
    model = linearModule(input_num)

    # 定义训练参数
    critarition = torch.nn.MSELoss()
    optimizer = SGD(params=model.parameters(), lr=0.0001)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    start_epoch = 0
    atale = 0
    best_mse = 10000
    step = 0
    patience = 5
    global_loss = []
    global_mse = []

    # 尝试训练
    # model, optimizer, scheduler, start_epoch = load_model(path="result/elo-merchant", model=model, optimizer=optimizer, scheduler=scheduler, accelerator=accelerator, epoch=0)

    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader)

    for epoch in range(start_epoch, 300):
        # 训练
        model.train()
        train_loss = 0
        train_bar = tqdm(
            train_loader, disable=not accelerator.is_local_main_process)
        if accelerator.is_local_main_process:
            train_bar.set_description(f'Epoch [{epoch + 1}/10] Training')

        for features, labels in train_bar:

            y_pred = model(features.float())
            loss = critarition(y_pred, labels.float())

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            if accelerator.is_local_main_process:
                train_loss = '{0:1.5f}'.format(loss)
                train_bar.set_postfix({'loss': f'{train_loss}'})
                writer.add_scalar('Train Loss', loss, step)
                step += 1
        # scheduler.step()
        global_loss.append(float(train_loss))

        # 验证
        model.eval()
        evaluator = evaluate.load("mse")
        val_bar = tqdm(
            val_loader, disable=not accelerator.is_local_main_process)
        if accelerator.is_local_main_process:
            val_bar.set_description(f'Epoch [{epoch + 1}/10] Validation')
        for features, labels in val_bar:
            with torch.no_grad():
                predictions = model(features.float())
            evaluator.add_batch(predictions=predictions,
                                references=labels.float())

        # 计算loss和acc并保存到tensorboard
        evaluate_result = evaluator.compute()
        accelerator.print(evaluate_result)

        global_mse.append(evaluate_result['mse'])

        accelerator.print(
            f"Epoch [{epoch + 1}/10] loss = {train_loss}, mse = {evaluate_result['mse']:.5f} %")

        if accelerator.is_local_main_process:
            writer.add_scalar('Val Acc', evaluate_result['mse'], epoch)

        # 保存模型
        if evaluate_result['mse'] < best_mse:
            best_mse = evaluate_result['mse']
            accelerator.print(f"Epoch [{epoch + 1}/10] 保存模型")
            # 保存模型
            save_model('elo -merchant', epoch, model,
                       optimizer, scheduler, accelerator)
            stale = 0
        # else:
        #     stale += 1
        #     if stale > patience:
        #         accelerator.print(f"连续的 {patience}  epochs 模型没有提升，停止训练")
        #         accelerator.end_training()
        #         break

    accelerator.print(f"最低MSE: {best_mse:.5f}")

    return global_loss, global_mse


def main():
    same_seeds(42)
    accelerator = Accelerator()
    write = SummaryWriter() if accelerator.is_local_main_process else None
    Logger(write.get_logdir() if write is not None else None)

    # train_path = 'data/all_train_features.csv'
    # test_path = 'data/all_test_features.csv'
    # train = pd.read_csv(train_path)

    train = pd.read_csv('data/deep-test.csv')
    train = np.array(train)
    train = np.nan_to_num(train)
    f = np.array(list(train)[0])

    # 1. 加载数据集合
    accelerator.print(f'加载数据集')
    train_loader, val_loader = loader_database(train)

    # 2. 训练模型
    y1, y2 = train_model(train_loader, val_loader, train.shape[1]-1, write)

    # 创建一个 8 * 6 点（point）的图，并设置分辨率为 80
    figure(figsize=(8, 6), dpi=80)

    # 创建一个新的 1 * 1 的子图，接下来的图样绘制在其中的第 1 块（也是唯一的一块）
    subplot(1, 1, 1)
    X = list(range(1, 301))
    y1 = list(y1)
    y2 = list(y2)
    plot(X, y1, color="blue", linewidth=1.0, linestyle="-")

   # 绘制正弦曲线，使用绿色的、连续的、宽度为 1 （像素）的线条
    plot(X, y2, color="green", linewidth=1.0, linestyle="-")

    plt.show()
    plt.savefig('sigmod-layer5.jpg')


if __name__ == "__main__":
    main()
