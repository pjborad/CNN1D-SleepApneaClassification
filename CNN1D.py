import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image  as mpimg
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam,RMSprop,SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix


x = pd.read_csv('ECG_2000_train.csv')
x = x.to_numpy()
x1 = pd.read_csv('ECG_2000_test.csv')
x1 = x1.to_numpy()
#x = np.expand_dims(x, axis=1)
y = pd.read_csv('ECG_2000_train_label.csv')
y = y.to_numpy() 
y = y.reshape(len(y),1) 
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray() 

y1 = pd.read_csv('ECG_2000_test_label.csv')
y1 = y1.to_numpy() 
y1 = y1.reshape(len(y1),1) 
y1 = ohe.fit_transform(y1).toarray() 

main = []
for i in range(len(x)):
    d = x[i]
    d = np.reshape(d,(6000,1))
    main.append(d)
main=np.array(main)
x = main

main1 = []
for i in range(len(x1)):
    d = x1[i]
    d = np.reshape(d,(6000,1))
    main1.append(d)
main1=np.array(main1)
x1 = main1
"""
for i in range(500):
  plt.subplot( 500// 20, 20, i + 1)
  view = X[i]
  plt.imshow(view*255,cmap=plt.cm.gray)
  plt.subplot(1,46,i+1)
  view = test_x[i+46]
  plt.imshow((view.reshape(40,40))*255,cmap=plt.cm.gray)
  plt.show()
"""

epochs =5
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0,random_state=42)
x_test,q,y_test,r=train_test_split(x1,y1,test_size=0,random_state=42)
#x_train= x
x_test = x1
#y_train = y
y_test = y1

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D(filters=8, kernel_size=15, strides=1,activation='relu', input_shape=(6000,1)))
#model.add(tf.keras.layers.Dropout(0.2))#covariateshift
model.add(tf.keras.layers.MaxPooling1D(pool_size=2,strides=2))
#model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=17, strides=1, activation='relu'))
#model.add(tf.keras.layers.Dropout(0.2))#covariateshift
model.add(tf.keras.layers.MaxPooling1D(pool_size=2,strides=2))
model.add(tf.keras.layers.BatchNormalization())
 
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=19,strides=1, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2,strides=2))
model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Flatten()) 
model.add(tf.keras.layers.Dense(1000, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(2, activation='sigmoid'))  
model.summary()

model.compile(optimizer=RMSprop(lr=0.01), loss='binary_crossentropy', metrics=['acc'])

history = model.fit(x_train,y_train,
                    epochs=epochs)
y_prob = model.predict(x_test)
predicted_y =np.argmax(y_prob, axis=1)
y_test = np.argmax(y_test,axis=1)
accuracy = accuracy_score(y_test,predicted_y)
print(accuracy)

c = confusion_matrix(y_test, predicted_y)
cd = pd.DataFrame(c)
print(cd)
#cd.to_excel('confusionMatrix.xlsx',index=False)



epochs_range = range(1,epochs+1)
plt.plot(epochs_range, history.history['acc'])
#plt.plot(epochs_range ,history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Acurracy')
plt.xlabel('Epochs')
plt.show()

plt.plot(epochs_range,history.history['loss'])
#plt.plot(epochs_range,history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()
