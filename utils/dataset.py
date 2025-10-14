import pandas as pd

def DP():
    target = 0
    data = pd.read_csv('./data/data_to_predict.csv', index_col=0)
    data_full = data[data['a01'] != 0].copy()
    # train =
    # val =
    # test = data[data['a01'] == 0].copy()

def get_data(datasetname):
    if datasetname == 'DP':
        return DP()
    else:
        print("There is no data.")

def preprocess(N_rows, parse_dates, file_name):
    with open(file_name, 'r') as f:
        total_rows = sum(1 for line in f)

    df = pd.read_csv(file_name, header=0, delimiter=',', nrows=N_rows, skiprows=0)
    return df.astype(float)

# time_index
def data_transform(df,encode_cols):
    # scaler = StandardScaler()
    # df = df.index
    # 对时间序列列进行标准化
    # df = scaler.fit_transform([df])
    for col in encode_cols:
        df[col] = df[col].astype('category')
    df = pd.get_dummies(df, columns=encode_cols)
    return df

def create_data(data, seq_len):
    N = len(data)
    print(N)
    X = []
    Y = []
    for i in range(N-seq_len-1):
        x = data[i:i+seq_len]
        X.append(x)
        y = data[i+seq_len]
        Y.append(y)
    return X, Y

def split_data(x, y, ratio):
    assert len(x)==len(y)
    N = len(x)
    train_x, test_x = x[:int(N*ratio)], x[int(N*ratio):]
    train_y, test_y = y[:int(N*ratio)], y[int(N*ratio):]
    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    DP()