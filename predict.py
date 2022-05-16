import requests
import json
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU,Bidirectional
from keras.models import load_model
import tensorflow as tf
from numpy import concatenate
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

# 转换成监督数据，四列数据，3->1，三组预测一组
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    # 将3组输入数据依次向下移动3，2，1行，将数据加入cols列表（技巧：(n_in, 0, -1)中的-1指倒序循环，步长为1）
    for i in range(n_in, 0, -1):
    	cols.append(df.shift(i))
    	names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    # 将一组输出数据加入cols列表（技巧：其中i=0）
    for i in range(0, n_out):
    	cols.append(df.shift(-i))
    	if i == 0:
    		names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    	else:
    		names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # cols列表(list)中现在有四块经过下移后的数据(即：df(-3),df(-2),df(-1),df)，将四块数据按列 并排合并
    agg = concat(cols, axis=1)
    # 给合并后的数据添加列名
    agg.columns = names
#     print(agg)
    # 删除NaN值列
    if dropnan:
    	agg.dropna(inplace=True)
    return agg

data=pd.read_csv("Stuttgartinfo.csv")
data=data.drop('Unnamed: 0',axis=1)

values = data.values
scaler = MinMaxScaler(feature_range=(0, 1))
values = scaler.fit_transform(values)

n_hours = 3
n_features = 3
reframed = series_to_supervised(values, n_hours, 1)

#print(reframed)
values = reframed.values
train = values[:125, :]
test = values[:, :]



n_obs = n_hours * n_features
# 有32=(4*8)列数据，取前24=(3*8) 列作为X，倒数第8列=(第25列)作为Y
train_X, train_y = train[:, :n_obs], train[:, -1]
test_X, test_y = test[:, :n_obs], test[:, -1]
#print(test_X.shape, len(test_X), test_y.shape)
# 将数据转换为3D输入，timesteps=3，3条数据预测1条 [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
#print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
model = tf.keras.models.load_model('./modelltsm/modelltsm')
model.summary()


yhat=model.predict(test_X)


test_Xx = test_X.reshape((test_X.shape[0], n_hours*n_features))
# 将预测列据和后7列数据拼接，因后续逆缩放时，数据形状要符合 n行*8列 的要求
inv_yhat = concatenate((test_Xx[:, -2:],yhat), axis=1)
# 对拼接好的数据进行逆缩放
#print(inv_yhat.shape)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]

test_yy = test_y.reshape((len(test_y), 1))
# 将真实列据和后7列数据拼接，因后续逆缩放时，数据形状要符合 n行*8列 的要求
inv_y = concatenate(( test_Xx[:, -2:],test_yy), axis=1)
# 对拼接好的数据进行逆缩放
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]
print("lstm_attention  R2:%.4f"%((r2_score(inv_yhat,inv_y ))))

pyplot.plot(inv_yhat,label='prediction')
pyplot.plot(inv_y,label='true')

pyplot.legend()
pyplot.show()

#print("lstm_attention  MSE:%.4f"%(mean_squared_error(inv_yhat,inv_y)))
#print("lstm_attention   MAE:%.4f"%(mean_absolute_error(inv_yhat,inv_y )))

