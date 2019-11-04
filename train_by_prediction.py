from numpy import loadtxt
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import pandas as pd
from IPython import embed
import pre_data as pr_d
import image_py as impy
import numpy as np
import matplotlib.pyplot as plt
import camera
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def convert_val_to_expression(value,expression):
    if value >= 0.9 :
        return expression
    if value>=0.5 and value <0.9:
        return "büyük ihtimal " + expression
    if value>=0.1 and value < 0.5:
        return "büyük ihtimal " + expression + " değil"
    if value<0.1:
        return expression + " değil"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

for item in range(1):
    camera.get_photo(item,"prediction_files/")



features=camera.load_prediction_images("prediction_files/*")
X =impy.convert_images_to_prediction_dataset(features)
Xi=np.array([x[1] for x in X]).astype(np.float32)
Xim=[x[0] for x in X]

try:
    # load model
    model = load_model('picture_model.h5')
    # summarize model.
    model.summary()
    # load dataset
except:
    inputShape=Xi.shape[1]
    ###MODEL SETTINGS
    model = Sequential([
        Dense(1024, activation='relu', input_shape=(inputShape,)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    adam=optimizers.Adam(learning_rate=0.0001)

    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# evaluate the model

score = model.predict(Xi)
scores=[[Xim[x],score[x]] for x in range(len(score))]
print("Train edilmemiş tahmin:")
for score in scores:
    print("İmaj: ",score[0],convert_val_to_expression(score[1],"Alper"))

cont=input("Devam etmek istiyor musunuz?(e/h):")
if cont=="h":
    exit()
Y=np.array([int(input("Bu imajın durumu")) for x in Xim])


hist = model.fit(Xi, Y, batch_size=32, epochs=100)
###MODEL TESTS
res=model.evaluate(Xi, Y)
model.save('picture_model.h5')
print("Accuracy:",res)
plt.plot(hist.history['loss'])
plt.plot(hist.history['accuracy'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Loss', 'Accuracy'], loc='upper right')
plt.show()
picture_score = model.predict(Xi)
picture_scores=[[Xim[x],picture_score[x]] for x in range(len(picture_score))]
print("Training sonrası tahmin")
for score in picture_scores:
    print("İmaj: ",score[0],convert_val_to_expression(score[1],"Alper"))
