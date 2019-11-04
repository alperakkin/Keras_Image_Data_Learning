from __future__ import print_function
import sys
import time
import cv2
from PIL import Image
from IPython import embed
import pandas as pd
import numpy as np
from glob import glob
camera_port = 0

def get_photo(item_no,location):
    filename=location+"/picture_"+str(item_no)+".png"
    camera = cv2.VideoCapture(camera_port)
    time.sleep(.1)
    return_value, image = camera.read()
    print("image",len(image),location)
    cv2.imwrite(filename, image)
    del(camera)
    return "picture_"+str(item_no)+".png"

def get_image():
    camera = cv2.VideoCapture(camera_port)
    time.sleep(.1)
    return_value, cv2_im = camera.read()
    del(camera)
    resized = cv2.resize(cv2_im, (32,32), interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    dataset=[]
    image_names=[]
    imageSize=None
    print("Tahminleme için resim sınıflandırması başlıyor...")
    pixels=[y for x in gray for y in x]
    imageSize=len(pixels)
    imgname="camera"
    dataset.append([imgname,pixels])
    return dataset,gray

def get_colors():
    camera = cv2.VideoCapture(camera_port)
    time.sleep(.1)
    return_value, cv2_im = camera.read()
    del(camera)
    resized = cv2.resize(cv2_im, (32,32), interpolation = cv2.INTER_AREA)
    embed()
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    dataset=[]
    image_names=[]
    imageSize=None
    print("Tahminleme için resim sınıflandırması başlıyor...")
    pixels=[y for x in gray for y in x]
    imageSize=len(pixels)
    imgname="camera"
    dataset.append([imgname,pixels])
    return dataset,gray


def load_training_images(image_path,data_path,label,scale=(32,32)):
    dataset=[]
    labels=[]
    dataframe=pd.read_csv(data_path)
    Image_List=glob(image_path)
    imageSize=None
    print("Resim sınıflandırması başlıyor...")
    for ind,img in enumerate(Image_List):

        im = cv2.imread(img)
        resized = cv2.resize(im, scale, interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        pixels=[y for x in gray for y in x]
        imageSize=len(pixels)
        imgname=str(img).split("image_list/train/")[1]
        labels.append(dataframe[dataframe['id']==imgname][label].values[0])
        dataset.append(pixels)
        print(str(ind)+"/"+str(len(Image_List))+ " tamamlandı.", end='\r')
        sys.stdout.flush()
    return dataset,labels

def load_prediction_images(image_path,scale=(32,32)):
    dataset=[]
    image_names=[]
    Image_List=glob(image_path)
    imageSize=None
    print("Tahminleme için resim sınıflandırması başlıyor...")
    for ind,img in enumerate(Image_List):
        im = cv2.imread(img)
        resized = cv2.resize(im, scale, interpolation = cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        pixels=[y for x in gray for y in x]
        imageSize=len(pixels)
        imgname=str(img).split("prediction_files/")[1]
        dataset.append([imgname,pixels])
        print(str(ind)+"/"+str(len(Image_List))+ " tamamlandı.", end='\r')
        sys.stdout.flush()
    return dataset

def show_image(img):
    cv2.imshow('image',img)
    cv2.waitKey(3000)
    input("Resmi kapatmak için bir tuşa basın")

get_colors()
