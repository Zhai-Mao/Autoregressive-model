import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
import xgboost as xgb
import operator
import plotly.graph_objects as go

def plot_future(historical_idx, train_y_np, historical_data, future_idx, future_predictions_original, future_steps, scale):
    # 创建图表
    fig = go.Figure()

    # 训练数据
    fig.add_trace(go.Scatter(
        x=historical_idx[:len(train_y_np)],
        y=scale.inverse_transform(historical_data[:len(train_y_np)].reshape(-1, 1)).flatten(),
        mode='lines',
        name='Training Data',
        line=dict(color='blue', width=1)
    ))

    # 测试数据
    fig.add_trace(go.Scatter(
        x=historical_idx[len(train_y_np):],
        y=scale.inverse_transform(historical_data[len(train_y_np):].reshape(-1, 1)).flatten(),
        mode='lines',
        name='Test Data',
        line=dict(color='green', width=1)
    ))

    # 未来预测
    fig.add_trace(go.Scatter(
        x=future_idx,
        y=future_predictions_original,
        mode='lines+markers',
        name='Future Prediction',
        line=dict(color='red', width=2),
        marker=dict(size=4)
    ))

    # 添加预测起始线
    fig.add_vline(x=len(historical_data), line_dash="dash", line_color="purple",
                  annotation_text="预测起点")

    fig.update_layout(
        title=f'RNN时间序列预测 - 未来{future_steps}步',
        xaxis_title='时间步',
        yaxis_title='数值',
        width=1200,
        height=600
    )

    fig.show()

def plot_test(x_values, combined_target, combined_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values,
        y=combined_target,
        mode='lines',
        name='Target',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=x_values,
        y=combined_pred,
        mode='lines',
        name='prediction',
        line=dict(color='red')
    ))

    fig.show()

def plot(x_values, combined_y):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_values,
        y=combined_y.flatten(),
        mode='lines',
        name='data_line'
    ))
    fig.show()

# plot
def plot_train_loss(train_loss, test_loss):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(train_loss))),
        y=train_loss,
        mode='lines',
        name='Train Loss',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=list(range(len(test_loss))),
        y=test_loss,
        mode='lines',
        name='Test Loss',
        line=dict(color='red')
    ))

    fig.update_layout(
        title='train and test loss line',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        showlegend=True
    )
    fig.show()



# 这部分有问题
def feature_importance_plot(importance_sorted, title):
    df = pd.DataFrame(importance_sorted, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()

    # 纯 matplotlib 画法
    # fig, ax = plt.subplots(figsize=(12, 10))
    # # print("feature is:", df['feature'])
    # # print("fscore is:", df['fscore'])
    # ax.barh(df['feature'], df['fscore'])
    # ax.set_title('XGBoost Feature Importance')
    # ax.set_xlabel('relative importance')
    # fig.tight_layout()
    #
    # out_file = fr'E:\DP\output\{title}.png'
    # fig.savefig(out_file, dpi=300)
    # print(f'[OK] 图片已保存到：{os.path.abspath(out_file)}')
    # plt.close(fig)  # 关键：彻底释放资源

def xgb_importance(df, test_ratio, xgb_params, ntree, early_stop, plot_title):
    df = pd.DataFrame(df)
    # 这里把时间序列作为Y，也就是预测下一个时间点的特征。
    Y = df.iloc[:, 0]
    X = df.iloc[:, 1:]
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=test_ratio,
                                                        random_state=42)
    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)

    watchlist = [(dtrain, 'train'), (dtest, 'validate')]
    xgb_model = xgb.train(xgb_params, dtrain, ntree,
                          evals=watchlist,
                          early_stopping_rounds=early_stop,
                          verbose_eval=True)

    importance = xgb_model.get_fscore()
    importance_sorted = sorted(importance.items(), key=operator.itemgetter(1))
    feature_importance_plot(importance_sorted, plot_title)