import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.dataset import *
from utils.visualize import *

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import xgboost as xgb
import operator

from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import numpy as np
import scipy.stats as st



# 使用前9000行
N_rows = 9000
# 日期
parse_dates = ['id']

file_name = "data/data_to_predict.csv"
# 时间特征编码
encode_cols = ['a10']


# preprocess
df = preprocess(N_rows, parse_dates, file_name)
# encode data
df = data_transform(df, encode_cols)

xgb_params = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    'subsample': 0.80,
    'colsample_bytree': 0.85,
    'eta': 0.1,
    'max_depth': 10,
    'seed': 42
}

val_ratio = 0.2
ntree = 300
early_stop = 50



print('-------Xgboost Using All Numeric Features------',
      '\n------inital model feature importence-----')

# fig_allFeatures = xgb_importance(
#     df, val_ratio, xgb_params, ntree, early_stop, 'ALL_Features'
# )
# plt.show()


##########################################################
#xgboost using only datatime information
# 生成未来时间序列数据骨架
# 这个是生成空白列的内容,创建只有一列a01全部为0的200行
def get_unseen_data(unseen_start, steps, encode_cols):
    # 生成从 unseen_start + 1 开始的连续id
    index = np.arange(unseen_start + 1, unseen_start + steps + 1)

    # 创建包含id列和a01列的数据框架
    df = pd.DataFrame({
        'id': index,  # 第一列：从37979开始逐渐增加
        'a01': np.zeros(steps) # 第二列：全为0
    })

    # print(df)
    return df

def xgb_data_split(df, unseen_start_date, steps, test_start_date, encode_cols):
    # 创建一个只有时间戳与a01列的空数据（未来数据）,包括训练的，测试的，还有未来的测试数据
    unseen = get_unseen_data(unseen_start_date, steps,
                             encode_cols)

    # 这个就将新的和旧的聚合在一起
    df = pd.concat([df, unseen], axis=0, ignore_index=True)
    print("new df is:",df)
    # 这里就没有a10，因为只有一列
    # df = data_transform(df, encode_cols)
    # print("new df is:", df)
    # 使用基于id列的值进行数据分割，而不是索引
    df_unseen = df[df['id'] > unseen_start_date].copy()
    df_test = df[(df['id'] > test_start_date) & (df['id'] <= unseen_start_date)].copy()
    df_train = df[df['id'] <= test_start_date].copy()

    return df_unseen, df_test, df_train

df = preprocess(N_rows, parse_dates, file_name)
# df = df['a01']

# 现在的df就只有一列，只有a01列这个列
df = df.iloc[:, 0:2]
df = pd.DataFrame(df)
print("df is:",df)

df.index[-1] # 37970

test_start_date = 37000
unseen_start_date = 37970
# 预测未来步数
steps = 200

df_unseen, df_test, df = xgb_data_split(
    df, unseen_start_date, steps, test_start_date, encode_cols
)
print('\n---------Xgboost on only datetime information-------------\n')

dim = {'train and validation data': df.shape,
       'test data': df_test.shape,
       'forecasting data': df_unseen.shape}
print(pd.DataFrame(list(dim.items()), columns=['Data', 'dimension']))

# train model
Y = df['a01']  # 目标变量：a01列
X = df[['id']]  # 特征：id列
# print(X)
X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=val_ratio, random_state=42
)
X_test = xgb.DMatrix(df_test[['id']])  # 测试集特征：id列
Y_test = df_test['a01']  # 测试集目标：a01列
X_unseen = xgb.DMatrix(df_unseen[['id']])  # 未来数据特征：id列
# print(df_unseen)

dtrain = xgb.DMatrix(X_train, y_train)
dval = xgb.DMatrix(X_val, y_val)
watchlist = [(dtrain, 'train'), (dval, 'validate')]

param_sk = {
    'objective': 'reg:squarederror',
    'subsample': 0.8,
    'colsample_bytree': 0.85,
    'seed': 42
}

skrg = XGBRegressor(**param_sk)
skrg.fit(X_train, y_train)

params_grid = {
    "n_estimators": st.randint(100, 500),
    'max_depth': st.randint(6, 30)
}

search_sk = RandomizedSearchCV(
    skrg, params_grid, cv=3, random_state=1, n_iter=10, verbose=2
)
search_sk.fit(X, Y)

print("best parameters:", search_sk.best_params_); print(
    "best score:", search_sk.best_score_
)

params_new = {**param_sk, **search_sk.best_params_}
model_final = xgb.train(params_new, dtrain, evals=watchlist,
                        early_stopping_rounds=early_stop, verbose_eval=True)
print('------------xgboost using datatime features only-----------',
      '\n----Grid search model feature importance------')

importance = model_final.get_fscore()
importance_sorted = sorted(importance.items(), key=operator.itemgetter(1))
fig1 = feature_importance_plot(importance_sorted, 'feature importance')
plt.show()

###########################################################################
# Forcasting
# prediction to testing data
# 预测并将预测的结果按照测试集的索引存放
Y_hat = model_final.predict(X_test)
Y_hat = pd.DataFrame(Y_hat, index=Y_test.index, columns=['predicted'])

# 预测未来数据
# X_unseen = xgb.DMatrix(df_unseen.iloc[:, 1:])
unseen_y = model_final.predict(X_unseen)
forecasts = pd.DataFrame(
    unseen_y, index=df_unseen.index, columns=["forecasts"]
)

# print(forecasts)

# forecast results using itinal model
xgb_model = xgb.train(xgb_params, dtrain, ntree, evals=watchlist,
                      early_stopping_rounds=early_stop, verbose_eval=False)
Y_hat = xgb_model.predict(X_test)
Y_hat = pd.DataFrame(Y_hat, index=Y_test.index, columns=["test_predicted"])
unseen_y = xgb_model.predict(X_unseen)
forecasts = pd.DataFrame(
    unseen_y, index=df_unseen.index, columns=["forecasts"]
)
print(forecasts)

# 绘制预测结果并保存图片
plt.figure(figsize=(12, 8))

# 方法1：简单折线图
plt.subplot(2, 2, 1)
plt.plot(forecasts.index, forecasts['forecasts'], 'b-', linewidth=2, label='预测值')
plt.title('未来预测结果')
plt.xlabel('索引')
plt.ylabel('预测值')
plt.legend()
plt.grid(True, alpha=0.3)



