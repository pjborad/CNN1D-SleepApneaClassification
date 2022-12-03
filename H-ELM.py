import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import scipy


ohe = OneHotEncoder()
scaler = MinMaxScaler()
train = pd.read_csv("logEntropy_balance.csv")
train = train.to_numpy()
train=scaler.fit_transform(train) #minmaxscalling

test = pd.read_csv("Label_balance.csv")
test = test.to_numpy()
test = ohe.fit_transform(test).toarray() #onehotencoding

x_train,x_test,y_train,y_test = train_test_split(train,test,test_size=0.15,random_state=42)

def input_to_hidden(x, Win, bia):
    a = np.dot(x, Win)
    a = a+bia
    return a
def activation(x):
    #b = np.maximum(x, 0, x) # ReLU
    b = np.tanh(x) #tanh
    return b
def inv_activation(x):
    b = np.arctanh(x)
    return b


HIDDEN_UNITS = 1600
W = np.random.normal(size=[x_train.shape[1], HIDDEN_UNITS])
B = np.random.normal(size=[HIDDEN_UNITS])
H = input_to_hidden(x_train,W,B)
H = activation(H)
print('shape of H',H.shape)
beta = np.dot(scipy.linalg.pinv(H),y_train)
print('shape of beta',beta.shape)

HIDDEN_UNITS1 = 3200
W1 = np.random.normal(size=[H.shape[1], HIDDEN_UNITS1])
B1 = np.random.normal(size=[HIDDEN_UNITS])
H1 = np.dot(y_train,scipy.linalg.pinv(beta))
print('shape of H1',H1.shape)
WHE =  np.dot(scipy.linalg.pinv(H),inv_activation(H1))
H2 = activation(np.dot(H,WHE))
H2[np.isnan(H2)] = 0
print(H2.shape)
beta_new = np.dot(scipy.linalg.pinv(H2),y_train)
y = np.dot(H2,beta_new)

def predictmy(x_test):
    a = np.dot(x_test,W)+B
    a = activation(a)
    b = np.dot(a,WHE)
    b = activation(b)
    c = np.dot(b,beta_new)
    return c
    
c = predictmy(x_test)

"""
correct = 0
total = y.shape[0]

for i in range(total):
    predicted = np.argmax(c[i])
    test = np.argmax(y_test[i])
    correct = correct + (1 if predicted == test else 0)
print('Accuracy: {:f}'.format(correct/total))
"""
y_prob = c
predicted_y = np.argmax(y_prob, axis=1) 
y_test1 = np.argmax(y_test, axis=1) 
c = confusion_matrix(y_test1, predicted_y) 
print(c)


"""
HIDDEN_UNITS1 = 700
Win1 = np.random.normal(size=[X.shape[1], HIDDEN_UNITS1])
bia1 = np.random.normal(size=[HIDDEN_UNITS1])
X1 = input_to_hidden(X,Win1,bia1)
Xt1 = np.transpose(X1)



Wout = np.dot(np.linalg.pinv(np.dot(Xt2, X2)), np.dot(Xt2, y_train))



x_final = np.dot(x_test,Win)+bia
x_final = np.dot(x_final,Win1)+bia1
print(x_final)
y = predict(x_final,Win2,bia2)




#y_prob = predict(x_test,Win1,bia1)
#predicted_y = np.argmax(y_prob, axis=1) 
#y_test1 = np.argmax(y_test, axis=1) 

#c = confusion_matrix(y_test1, predicted_y) 
#print(c)
"""