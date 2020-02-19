#!usr/bin/python
# -*- coding: utf-8 -*-
#author:ly
import numpy as np
from keras import backend as K
from keras import Input, Model, Sequential
from keras.layers import Dense, BatchNormalization, Activation, regularizers
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.decomposition import PCA
from sklearn import tree, preprocessing, metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

print("读取数据中.........")
train_X=np.load('C:\\Users\\Administrator\\Desktop\\No4\\一维npy\\train.npy')
test_X=np.load('C:\\Users\\Administrator\\Desktop\\No4\\一维npy\\test.npy')
train_y = np.loadtxt(r"C:\Users\Administrator\Desktop\No4\一维npy\train_y.txt",dtype=str)
test_y = np.loadtxt(r"C:\Users\Administrator\Desktop\No4\一维npy\test_y.txt",dtype=str)
print("读取成功........")
train_y = np.array(train_y)
print(train_y.shape,test_y.shape)
total_X = np.vstack((train_X,test_X))

train_y = train_y.reshape(train_y.shape[0], -1)
test_y = test_y.reshape(test_y.shape[0], -1)
data_y = np.vstack((train_y, test_y))


min_max_scaler = preprocessing.MinMaxScaler()
X_total_minmax = min_max_scaler.fit_transform(total_X)

#加入高斯噪声
noise_factor = 0.5
x_train_noisy = X_total_minmax + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_total_minmax.shape)
all_X = np.clip(x_train_noisy, 0., 1.)

X_train, X_test, y_train, y_test = train_test_split(all_X, data_y, test_size=0.4, random_state=42)
#train_size = train_X.shape[0]
#X_train = X_total_minmax[0:train_size,:]
#X_test = X_total_minmax[train_size:,:]
#print(X_total_minmax)
'''
# y值one-hot
encoder = LabelEncoder()
encoder.fit(y_train)
y_train_hot = encoder.transform(y_train)
y_train = np_utils.to_categorical(y_train,num_classes=4)
encoder.fit(test_y)
y_test = encoder.transform(test_y)
y_test = np_utils.to_categorical(y_test,num_classes=4)
'''
encoding_dam = 40
input_img=Input(shape=(16384,))

#建立编码层
encoded = Dense(1000,activation='relu')(input_img)
encoded = Dense(256 ,activation='relu')(encoded)
encoded = Dense(128,activation='relu')(encoded)
encoder_output = Dense(encoding_dam,name='mid')(encoded)

#建立解码层
decoded = Dense(128,activation='relu')(encoder_output)
decoded = Dense(256,activation='relu')(decoded)
decoded = Dense(1000,activation='relu')(decoded)
decoded = Dense(16384,activation='sigmoid')(decoded)

# 构建自编码模型
autoencoder = Model(inputs=input_img, outputs=decoded)

# 构建编码模型
encoder = Model(inputs=input_img, outputs=encoder_output)

#编译模型
autoencoder.compile(optimizer='adam',loss='mse')

autoencoder.summary()
# training
autoencoder.fit(X_train, X_train, epochs=10, batch_size=256, shuffle=True)

auto_trainX=encoder.predict(X_train)
auto_testX = encoder.predict(X_test)
print(auto_trainX.shape)
print(auto_testX.shape)

#使用决策树
print("Training---------- 决策树")
clf = tree.DecisionTreeClassifier()
clf.fit(auto_trainX,y_train)
clf_pre = clf.predict(auto_testX)
score = metrics.accuracy_score(y_test,clf_pre)
score1 = clf.score(auto_testX,y_test)
rec = classification_report(y_test,clf_pre)
print(score,score1)
print(rec)

#使用BP网络
print("Training---------- BP")
bp_c = MLPClassifier(solver='lbfgs', alpha=0.00001, hidden_layer_sizes=(5, 5, 4), activation='relu')
bp_c.fit(auto_trainX,y_train)
bp_pre = bp_c.predict(auto_testX)
bp_score=metrics.accuracy_score(y_test,bp_pre)
bp_rec = classification_report(y_test,bp_pre)
print( bp_score)
print(bp_rec)
#逻辑回归
print("Training---------- 逻辑回归")
logist = LogisticRegression()
logist.fit(auto_trainX,y_train)
#logist_score = logist.score(Pca_test,y_test)
logist_pre = logist.predict(auto_testX)
logist_score=metrics.accuracy_score(y_test,logist_pre)
logist_rec = classification_report(y_test,logist_pre)
print( logist_score)
print(logist_rec)
'''
logist_f=logist_pre==test_y
logist_t=test_y.shape[0]-0.5*np.count_nonzero(logist_f == False)
logist_score=logist_t/test_y.shape[0]
print(logist_score)
'''
#支持向量机
print("Training---------- SVM")
svc = SVC()
svc.fit(auto_trainX,y_train)
predict = svc.predict(auto_testX)
svc_score = metrics.accuracy_score(y_test, predict)
svc_rec = classification_report(y_test,predict)
print(svc_score)
print(svc_rec)

plt.figure()
namelist=['tree','BP','logi','SVM']
numlist = [score,bp_score,logist_score,svc_score]
plt.bar(range(len(namelist)), numlist, color='rgby')
index=[0,1,2,3,]
plt.xticks(index, namelist)
plt.show()



'''
#keras BP神经网络
print("Training---------- Keras Bp网络")
model = Sequential()
model.add(Dense(units=16,input_dim=encoding_dam,activation='relu'))
model.add((Dense(units=8,activation='relu')))

model.add(Dense(units=4,activation='sigmoid'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print("\n Training -------- ")
model.fit(auto_trainX,y_train, epochs=50, batch_size=64)
print("\n Testing -------- ")
loss, accuracy = model.evaluate(auto_testX, y_test)
print("test loss:", loss)
print("test accuracy:", accuracy)
'''

'''
# keras
model = Sequential([
    Dense(128, input_dim=n),
    BatchNormalization(),
    Activation('relu'),
    Dense(64),
    BatchNormalization(),
    Activation('relu'),
    Dense(16),
    BatchNormalization(),
    Activation('relu'),
    Dense(4),
    BatchNormalization(),
    Activation('softmax')
])

rsm = Adam(lr=0.01,epsilon=1e-8,decay=0.0)
model.compile(optimizer=rsm,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Training -------")

# 训练模型
model.fit(Pca_train,y_train, epochs=100, batch_size=32)

# 测试
print("\n Testing -------- ")
loss, accuracy = model.evaluate(Pca_test, y_test)
print("test loss:", loss)
print("test accuracy:", accuracy)
'''




