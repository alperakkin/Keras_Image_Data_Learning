import image_py as impy
import camera
import pandas as pd


def to_csv(purpose,iteration):
    csv_file =pd.read_csv("template.csv")
    for t in range(iteration):
        picture=camera.get_photo(str(t),"image_list/"+purpose+"/")
        csv_file.at[t,'id']=picture
        csv_file.at[t,'val']=str(input("Alper is that you?"))
        csv_file.to_csv("train.csv",index=False)

to_csv("train",30)
