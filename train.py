import argparse
import os
from config.default import cfg, update_cfg_from_args
# from config.default import get_config
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
import numpy as np
# import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn

from utils.dataset import create_data, split_data
from model.model import Simple_RNN, train_model, predict_future
from utils.visualize import plot_train_loss, plot, plot_test, plot_future

parser = argparse.ArgumentParser(description='tabluar regression')
# parser.add_argument('-c', '--config', type=str, required=False, default='config/DP.yaml', metavar="FILE", help='Path to config file')
parser.add_argument('-d', '--data_path', type=str, required=False, default='data/data_to_predict.csv', metavar="FILE",
                    help='Path to data')

parser.add_argument('--nrows', type=int, default=9888,  help='data line')
parser.add_argument('--step', type=int, default=10,  help='step')
parser.add_argument('--usecols', type=int, default=1,  help='usecols')
parser.add_argument('--future_steps', type=int, default=15,  help='future_steps')

parser.add_argument('--hid_size', type=int, default=3,  help='model_RNN hidden size')
parser.add_argument('--num_layers', type=int, default=1,  help='model_RNN NUM LAYER')

parser.add_argument('--lr', type=float, default=0.01,  help='learning rate')
parser.add_argument('--epochs', type=int, default=500,  help='epoch number of train')
args = parser.parse_args()
# config = get_config(args)

if __name__ == '__main__':
    # 用命令行参数更新配置
    cfg = update_cfg_from_args(cfg, args)

    # 冻结配置，防止后续修改
    cfg.freeze()

    print("最终配置:")
    print(cfg)

    # file_name = "data/data_to_predict.csv"
    data = pd.read_csv(args.data_path, usecols=[args.usecols], nrows=args.nrows)
    # print("data is:", data)

    scale = MinMaxScaler(feature_range=(0, 1))
    data = scale.fit_transform(data)
    x, y = create_data(data, args.step)

    x, y = np.asarray(x), np.asarray(y)
    print("x is:", x.shape)
    print("y is:", y.shape)
    x.reshape(-1, 10).shape
    # print(x.shape)
    train_x, train_y, test_x, test_y = split_data(x, y, 0.7)

    # plot
    combined_y = np.vstack((train_y, test_y))
    x_values = np.arange(len(combined_y))
    plot(x_values, combined_y)

    train_x = torch.from_numpy(train_x.astype(np.float32))
    train_y = torch.from_numpy(train_y.astype(np.float32))
    test_x = torch.from_numpy(test_x.astype(np.float32))
    test_y = torch.from_numpy(test_y.astype(np.float32))

    # print("x shape is:",x.shape)  #x shape is: (9989, 10, 1)
    in_size = x.shape[-1]
    out_size = in_size
    hid_size = args.hid_size
    num_layers = args.num_layers

    model = Simple_RNN(in_size, hid_size, out_size, num_layers)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fun = nn.MSELoss()

    train_loss, test_loss = train_model(model,
                                        loss_fun,
                                        optimizer,
                                        train_x,
                                        test_x,
                                        train_y,
                                        test_y,
                                        epochs=args.epochs)



    plot_train_loss(train_loss, test_loss)


    # plot test prediction
    train_y_np  = train_y.detach().numpy()
    test_y_np = test_y.detach().numpy()
    combined_target = np.vstack((train_y_np, test_y_np)).flatten()
    test_pred = model(test_x).detach().numpy()
    combined_pred = np.vstack((train_y_np, test_pred)).flatten()

    x_values = np.arange(len(combined_target))
    plot_test(x_values, combined_target, combined_pred)

    future_steps = args.future_steps
    last_sequence = test_x[-1:].clone()  # shape:[1,10,1]
    print(f"用于预测的最后序列形状：{last_sequence.shape}")

    # 反归一化到原始尺度

    # 未来预测
    future_predictions = predict_future(model, last_sequence, future_steps)
    print(f"未来预测完成")

    # 反归一化到原始尺度
    future_predictions_original = scale.inverse_transform(future_predictions.reshape(-1, 1)).flatten()

    # 可视化结果
    train_y_np = train_y.detach().numpy()
    test_y_np = test_y.detach().numpy()
    historical_data = np.vstack((train_y_np, test_y_np)).flatten()

    # 创建时间索引
    historical_idx = np.arange(len(historical_data))
    future_idx = np.arange(len(historical_data), len(historical_data) + len(future_predictions))
    plot_future(historical_idx, train_y_np, historical_data, future_idx, future_predictions_original, future_steps, scale)


    # 保存预测结果
    future_df = pd.DataFrame({
        'step': range(1, future_steps + 1),
        'normalized_prediction': future_predictions,
        'original_prediction': future_predictions_original
    })
    future_df.to_csv('future_predictions.csv', index=False)
    print("预测结果已保存到 future_predictions.csv")

    # 打印预测统计
    print(f"\n预测结果统计:")
    print(f"预测步数: {future_steps}")
    print(f"归一化尺度 - 最小值: {future_predictions.min():.6f}, 最大值: {future_predictions.max():.6f}")
    print(f"原始尺度 - 最小值: {future_predictions_original.min():.6f}, 最大值: {future_predictions_original.max():.6f}")
    print(f"前10个预测值: {future_predictions_original[:10]}")