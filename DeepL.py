import pandas as pd 
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras import regularizers
from keras.layers import LeakyReLU

import random

def bank(path, val_ratio):
    df=pd.read_csv(path)
    ls = df.education.unique().tolist()
    enc_ed = np.array(list(map(lambda x:ls.index(x),df.education.tolist()))).reshape(-1,1)
    enc_h = np.array(list(map(lambda x:0 if x=='yes' else 1,df.housing.tolist()))).reshape(-1,1)
    y = np.array(list(map(lambda x: 1 if x=='nonexistent' else (2 if x=='success' else 3),df.poutcome.tolist()))).reshape(-1,1)
    x = np.hstack((enc_ed,enc_h))
    idx = list(range(x.shape[0]))
    random.shuffle(idx)
    valamount = int(x.shape[0]*val_ratio)
    #import pdb;pdb.set_trace()
    xtrain = x[idx][valamount:]
    xtest = x[idx][:valamount]
    ytrain = y[idx][valamount:]
    ytest = y[idx][:valamount]
    model = Sequential()
    model.add(Dense(10,kernel_initializer = keras.initializers.he_uniform(seed=None),input_dim = xtrain.shape[1],kernel_regularizer=regularizers.l2(0.01)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
    model.add(Dense(1))
    optimizer=keras.optimizers.Adagrad(lr=0.05, epsilon=None, decay=0.0)
    model.compile(loss='mse', optimizer=optimizer,metrics=['accuracy'])
    model.fit(xtrain, ytrain, batch_size=1000, epochs=1000, verbose=2,validation_split=0.1)   
    pred = model.predict(xtest)
    return pred
    
path = 'D:\\Executable\\banking.csv'
pred = bank(path,0.1)
print(pred)