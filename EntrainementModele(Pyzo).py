from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image

import numpy as np

nbpixel = 300*600*3
nbzones=10

model = keras.Sequential()
model.add(layers.Conv2D(5,3,  activation="relu"))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(1,3,  activation="relu"))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(nbpixel,activation="relu"))
model.add(layers.Dense(2,activation="softmax"))



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

for i in range(100):
    img=Image.open(pathimg+"img{}.png".format(i))
    Lx.append(np.array(img))

    info=open(pathinfo+"Info{}.txt".format(i))
    reg=getReg(info)
    info.close()
    indice=findelt(reg,Lreg)
    Ly.append(transfo(indice,12))


Lx=np.array(Lx,dtype="float64")
Ly=np.array(Ly)

model.compile()
model.fit(Lx,Ly)
##
z=model.predict(Lx[0])
print(z)
print(Ly[0])
