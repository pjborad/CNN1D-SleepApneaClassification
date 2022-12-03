
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

ohe = OneHotEncoder()
scaler = MinMaxScaler()



x = pd.read_csv("MIT_sample.csv")
x = x.to_numpy()
x = scaler.fit_transform(x)


y = pd.read_csv("MIT_label.csv")
y = y.to_numpy()
y = ohe.fit_transform(y).toarray()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20)
HIDDEN_UNITS = 1000
Win = np.random.normal(size=[x_train.shape[1], HIDDEN_UNITS])  
print(Win.shape) 
bia = np.random.normal(size=[HIDDEN_UNITS])
print(bia.shape)

def input_to_hidden(x):
    a = np.dot(x, Win)
    a = a+bia
    b = np.maximum(a, 0, a) # ReLU
    #b = np.tanh(a)
    return b

X = input_to_hidden(x_train)
print(X.shape)
Xt = np.transpose(X)

#Wout = np.dot(np.linalg.pinv(np.dot(Xt, X)), np.dot(Xt, y_train))
Wout = np.dot(np.linalg.pinv(X),y_train)
print(Wout.shape)



def predict(x):
    x = input_to_hidden(x)
    y = np.dot(x, Wout)
    return y

y = predict(x_test)
correct = 0
total = y.shape[0]
for i in range(total):
    predicted = np.argmax(y[i])
    test = np.argmax(y_test[i])
    correct = correct + (1 if predicted == test else 0)
print('Accuracy: {:f}'.format(correct/total))

y_prob = predict(x_test)
predicted_y = np.argmax(y_prob, axis=1) 
y_test1 = np.argmax(y_test, axis=1) 

c = confusion_matrix(y_test1, predicted_y) 
print(c)





