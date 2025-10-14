import torch
import torch.nn as nn

import numpy as np

class Simple_RNN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.layer = num_layers

        self.rnn = nn.RNN(self.in_dim, self.hid_dim, self.layer, nonlinearity='tanh', batch_first=True)
        self.fc = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer, x.size(0), self.hid_dim)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:,-1,:])
        return out

def train_model(model, criterion, optimizer, x_train, x_test, y_train, y_test, epochs=500):
    train_loss = np.zeros(epochs)
    test_loss = np.zeros(epochs)

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x_train)
        error = criterion(pred, y_train)

        error.backward()

        optimizer.step()

        train_loss[epoch] = error.item()

        test_pred = model(x_test)
        test_error = criterion(y_test, test_pred)
        test_loss[epoch] = test_error.item()

        if (epoch+1) % 5 ==0:
            print('Epoch : {} Train Loss: {}, test loss: {}'.format((epoch+1)/epochs, error.item(), test_error.item()))

    return train_loss, test_loss

# prediction future
def predict_future(model, last_sequence, future_steps):
    model.eval()
    predictions = []
    current_sequence = last_sequence.clone()

    print(f"开始预测未来 {future_steps} 步数据")
    print(f"初始序列形状：{current_sequence.shape}")

    with torch.no_grad():
        for i in range(future_steps):
            # 使用当前序列预测下一步
            next_pred = model(current_sequence)
            pred_value = next_pred.item()
            predictions.append(pred_value)

            # 更新序列：移除第一个时间步，添加新的预测值
            current_sequence = torch.cat([
                current_sequence[:, 1:, :], #移除第一个时间步
                next_pred.unsqueeze(-1)  #添加新的预测值，保持形状一致
            ], dim=1)

            # 打印进度
            if (i + 1) % 50 == 0:
                print(f"预测进度： {i + 1}/{future_steps}")

    return np.array(predictions)