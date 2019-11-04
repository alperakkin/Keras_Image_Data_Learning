from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import pandas as pd
from IPython import embed
import pre_data as pr_d
import image_py as impy
import camera
from numpy import loadtxt
from tensorflow.keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
###DATA SETTINGS####


features,labels=camera.load_training_images("image_list/train/*","train.csv","val")
X_train,X_val,X_test,Y_train,Y_val,Y_test =impy.convert_images_to_training_dataset(features,labels)


inputShape=X_train.shape[1]
###MODEL SETTINGS
try:
    # load model
    model = load_model('picture_model.h5')
    # summarize model.
    model.summary()
    # load dataset
    print("Model is loading")
except:
    ###MODEL SETTINGS
    print("New Model is creating")
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


###MODEL TRAINING
hist = model.fit(X_train, Y_train,
          batch_size=32, epochs=100,
          validation_data=(X_val, Y_val))


###MODEL TESTS
res=model.evaluate(X_test, Y_test)[1]

print(res)
#
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()


model.save('picture_model.h5')

####MODEL PREDICTIONS #####
# df_pred = pr_d.read_data('test.csv')
# pred_ds=pr_d.prepare_prediction_dataset(df_pred,features)
# result_set=model.predict(pred_ds)
# result_df=pd.DataFrame()
# result_df['PassengerId']=df_pred['PassengerId']
# result_df['Survived']=result_set.round().astype('int32')
# result_df.to_csv("result.csv",index = False)
