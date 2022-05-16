from django.shortcuts import redirect
from flask import Flask
from flask import render_template
from flask import request
from flask import Response
import json
import getdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import io
import random
import base64
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



app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1 #Prevent caching, so image can be updated

company_list = ["50_Hertz", "Amprioin","TenneTTSO","TransnetBW"]
location_list = ["Brandenburg","Dortmund","Bayreuth","Stuttgart"]
form_data = []

#requests.args['variable name from html'] or .forms

@app.route("/")
def WelcomePage():
    BuoyIDs = 1
    DisplayData =2
    get_plot()
    
    # output = io.BytesIO()
    # FigureCanvas(fig).print_png(output)
    # # return Response(output.getvalue(), mimetype='image/png')
    # img = "data:image/png;base64,"
    # img += base64.b64encode(output.getvalue()).decode('utf8')


    return render_template("Website.html",company_list = company_list)
def get_plot():
    # fig = Figure()
    # axis = fig.add_subplot(1, 1, 1)
    # data = getdata.get_wind()
    # axis.plot(data)
    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]
    plt.plot(xs, ys)
    plt.savefig('static/plot.png')
    
    print("A")
    # return fig

@app.route("/", methods=['GET','POST'])
def get_wind():
    if request.method == 'GET':
        return f"The URL /data is accessed directly. Try going to '/form' to submit form"
    if request.method == 'POST':
        form_data = request.form.getlist('company')
        #####
        #Insert functions here

        data = pd.read_csv(form_data[0]+".csv")
        data = data.drop('Unnamed: 0', axis=1)

        values = data.values
        scaler = MinMaxScaler(feature_range=(0, 1))
        values = scaler.fit_transform(values)

        n_hours = 3
        n_features = 3
        reframed = series_to_supervised(values, n_hours, 1)

        # print(reframed)
        values = reframed.values
        train = values[:125, :]
        test = values[:, :]

        n_obs = n_hours * n_features
        # 有32=(4*8)列数据，取前24=(3*8) 列作为X，倒数第8列=(第25列)作为Y
        train_X, train_y = train[:, :n_obs], train[:, -1]
        test_X, test_y = test[:, :n_obs], test[:, -1]
        # print(test_X.shape, len(test_X), test_y.shape)
        # 将数据转换为3D输入，timesteps=3，3条数据预测1条 [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
        test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
        # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
        model = tf.keras.models.load_model('./modelltsm_t/modelltsm_t')
        model.summary()

        yhat = model.predict(test_X)

        test_Xx = test_X.reshape((test_X.shape[0], n_hours * n_features))
        # 将预测列据和后7列数据拼接，因后续逆缩放时，数据形状要符合 n行*8列 的要求
        inv_yhat = concatenate((test_Xx[:, -2:], yhat), axis=1)
        # 对拼接好的数据进行逆缩放
        # print(inv_yhat.shape)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:, -1]

        test_yy = test_y.reshape((len(test_y), 1))
        # 将真实列据和后7列数据拼接，因后续逆缩放时，数据形状要符合 n行*8列 的要求
        inv_y = concatenate((test_Xx[:, -2:], test_yy), axis=1)
        # 对拼接好的数据进行逆缩放
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:, -1]
        print(len(inv_y))
        print("lstm_attention  R2:%.4f" % ((r2_score(inv_yhat, inv_y))))
        ###
        pyplot.clf()

        fig, axs = plt.subplots(2, 2)
        axs[0,0].plot(inv_yhat[0:int(len(inv_y)/4)], label='prediction')
        axs[0,0].plot(inv_y[0:int(len(inv_y)/4)], label='true')
        axs[0,0].set_title('Q1')

        axs[0,1].plot(inv_yhat[int(len(inv_y)/4)+1:int(len(inv_y)/2)], label='prediction')
        axs[0,1].plot(inv_y[int(len(inv_y)/4)+1:int(len(inv_y)/2)], label='true')
        axs[0,1].set_title('Q2')


        axs[1,0].plot(inv_yhat[int(len(inv_y)/2)+1:int(len(inv_y)/2+len(inv_y)/4)], label='prediction')
        axs[1,0].plot(inv_y[int(len(inv_y)/2)+1:int(len(inv_y)/2+len(inv_y)/4)], label='true')
        axs[1,0].set_title('Q3')


        axs[1,1].plot(inv_yhat[int(len(inv_y)*3/4)+1:], label='prediction')
        axs[1,1].plot(inv_y[int(len(inv_y)*3/4)+1:], label='true')
        axs[1,1].set_title('Q4')
        pyplot.legend()
        fig.tight_layout()
        pyplot.savefig('static/plot.png')
        ####
        return render_template('Website.html',form_data = form_data,company_list = company_list)
    return
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


# @app.route("/submitform", methods=['GET','POST'])
# def submit():
#     if request.method == 'GET':
#         return f"The URL /data is accessed directly. Try going to '/form' to submit form"
#     if request.method == 'POST':
#         form_data = request.form
#         #Insert functions here
#         return redirect("/")
#         # return render_template('Website.html',form_data = form_data,company_list = company_list)
#     return

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    if 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'no-store'
    return response

if __name__ == '__main__':
    #app.run(host="192.168.0.211", port="8090", threaded=True, debug=True, use_reloader=False)
    # app.run(host="0.0.0.0", port="80", threaded=True, debug=False, use_reloader=False)
    #app.run(host="118.139.75.207", port="8090", threaded=True, debug=True, use_reloader=False)
    app.run(host="0.0.0.0", port="8090", threaded=True, debug=True, use_reloader=False)

