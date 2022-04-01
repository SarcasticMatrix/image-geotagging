from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

import numpy as np


model = keras.Sequential()
model.add(layers.Conv2D(5,3,  activation="relu"))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(1,3,  activation="relu"))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(50,activation="relu"))
model.add(layers.Dense(12,activation="softmax"))



def getReg(file):
    """obtient la région d'un file InfoX.txt"""
    K= 2 # 2 pour région, 3 pour dep
    lines=file.readlines()
    file.close()

    Goodlines=[]
    for line in lines:
        if "formatted" in line:
            Goodlines.append(line)


    Good=Goodlines[-K].split(":")[1].split(",")[0][2:]
    return Good

def findelt(elt,List):
    """renvoie l'indice de la première occurence de elt dans List"""
    for indice,x in enumerate(List):
        if x==elt:
            return indice

    raise ValueError # string not in List

Lreg=[
"Auvergne-RhÃ´ne-Alpes",
"Bourgogne-Franche-ComtÃ©",
"Brittany",
"Centre-Val de Loire",
"Grand Est",
"Hauts-de-France",
"Normandy",
"Nouvelle-Aquitaine",
"Occitanie",
"Pays de la Loire",
"Provence-Alpes-CÃ´te d'Azur",
"ÃŽle-de-France"]

Lreg2=[
"Auvergne-Rhône-Alpes",
"Bourgogne-Franche-Comté",
"Brittany",
"Centre-Val de Loire",
"Grand Est",
"Hauts-de-France",
"Normandy",
"Nouvelle-Aquitaine",
"Occitanie",
"Pays de la Loire",
"Provence-Alpes-Côte d'Azur",
"Île-de-France"]


path="D:\\Travail\\TSP\\Cours\\PRO3600\\PRO3600_DATA\\"
pathimg=path+"\\Images\\"
pathinfo=path+"\\Infos\\"


def transfo(i,k):
    L=[0]*k
    L[i]=1
    return L

Lx=[]
Ly=[]

for i in range(1):
    img=Image.open(pathimg+"img{}.png".format(i))
    Lx.append(np.array(img))

    info=open(pathinfo+"Info{}.txt".format(i))
    reg=getReg(info)
    info.close()
    indice=findelt(reg,Lreg)
    Ly.append(transfo(indice,12))


print("hello")
Lx=np.array(Lx,dtype="float64")
Ly=np.array(Ly)

model.compile(loss="MeanSquaredError",optimizer="SGD")
model.fit(Lx,Ly)
##

print(Ly.shape)
print(Lx.shape)
##

model.compile(loss="MeanSquaredError",optimizer="SGD")
model.fit(Lx,Ly)

model.save("D:\\Travail\\Mymodel.h5")
##
import tensorflowjs as tfjs
model.compile(loss="MeanSquaredError",optimizer="SGD")
model.fit(Lx,Ly)

tfjs.converters.save_keras_model(model,"D:\\Perso\\Test Informatique\\Js Tests\\MyModel")


##
from keras.models import load_model
model_new = load_model("D:\\Travail\\Mymodel.h5")


model_new.predict(Lx)
