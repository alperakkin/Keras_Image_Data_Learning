from numpy import loadtxt
from tensorflow.keras.models import load_model
import os
from datetime import datetime
import camera
import image_py as impy
import numpy as np
import time
from IPython import embed

def convert_val_to_expression(value,expression):
    if value >= 0.9 :
        return "Merhaba "+expression +" " + str(value)
    if value>=0.5 and value <0.9:
        return "Sen " + expression + " misin? " + str(value)
    if value>=0.1 and value < 0.5:
        return "Seni tanıyamadım. "+expression +"  olabilir misin? " + str(value)
    if value<0.1:
        return "Sen "+expression + " değilsin " + str(value)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load model
model = load_model('picture_model.h5')
# summarize model.
model.summary()
# load dataset

# evaluate the model

input("hazırsan bir tuşa bas:")
#time.sleep(5)
print("Başladı")
start=datetime.now()
for item in range(1):
    features,orginal_image=camera.get_image()
X =impy.convert_images_to_prediction_dataset(features)
Xi=np.array([x[1] for x in X]).astype(np.float32)
Xim=[x[0] for x in X]



score = model.predict(Xi)
scores=[[Xim[x],score[x]] for x in range(len(score))]
print("Train edilmemiş tahmin:")
for score in scores:
    print("İmaj: ",score[0],convert_val_to_expression(score[1],"Alper"))

end=datetime.now()

result=float((end-start).microseconds/1000000)+(end-start).seconds


print("İşlem süresi: ",result,"saniyedir")
camera.show_image(orginal_image)
