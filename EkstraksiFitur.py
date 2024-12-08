import pandas as pd
import numpy as np
import cv2
import os
import csv
from scipy.stats import skew

#kondisi kolam jernih
path = "C:\dataset\EkstraksiFitur\outputresize\jernih"
data = os.listdir(path)
WarnaR = []
WarnaG = []
WarnaB = []
WarnaH = []
WarnaS = []
WarnaV = []
Kondisi = []

for gbr in data:
    gbr_read = cv2.imread(os.path.join(path, gbr))
    gbr_rgb = cv2.cvtColor(gbr_read, cv2.COLOR_BGR2RGB)
    (R, G, B) = cv2.split(gbr_rgb)
    meanR = np.mean(R)
    WarnaR.append(meanR)
    meanG = np.mean(G)
    WarnaG.append(meanG)
    meanB = np.mean(B)
    WarnaB.append(meanB)
    
    gbr_hsv = cv2.cvtColor(gbr_read, cv2.COLOR_BGR2HSV)
    H = gbr_hsv[:,:,0]
    S = gbr_hsv[:,:,1]
    V = gbr_hsv[:,:,2]
    meanH = np.mean(H)
    WarnaH.append(meanH)
    meanS = np.mean(S)
    WarnaS.append(meanS)
    meanV = np.mean(V)
    WarnaV.append(meanV)
    
    keterangan = '2'
    Kondisi.append(keterangan)
    
data1 = pd.DataFrame (WarnaR, columns=['Warna R'])
data2 = pd.DataFrame (WarnaG, columns=['Warna G'])
data3 = pd.DataFrame (WarnaB, columns=['Warna B'])
data4 = pd.DataFrame (WarnaH, columns=['Warna H'])
data5 = pd.DataFrame (WarnaS, columns=['Warna S'])
data6 = pd.DataFrame (WarnaV, columns=['Warna V'])
data7 = pd.DataFrame (Kondisi, columns=['Kondisi'])

listdata = [data1, data2, data3, data4, data5, data6, data7]
gabung = pd.concat(listdata, axis=1)

#kondisi kolam keruh
path = "C:\dataset\EkstraksiFitur\outputresize\keruh"
data = os.listdir(path)
WarnaR = []
WarnaG = []
WarnaB = []
WarnaH = []
WarnaS = []
WarnaV = []
Kondisi = []

for gbr in data:
    gbr_read = cv2.imread(os.path.join(path, gbr))
    gbr_rgb = cv2.cvtColor(gbr_read, cv2.COLOR_BGR2RGB)
    (R, G, B) = cv2.split(gbr_rgb)
    meanR = np.mean(R)
    WarnaR.append(meanR)
    meanG = np.mean(G)
    WarnaG.append(meanG)
    meanB = np.mean(B)
    WarnaB.append(meanB)
    
    gbr_hsv = cv2.cvtColor(gbr_read, cv2.COLOR_BGR2HSV)
    H = gbr_hsv[:,:,0]
    S = gbr_hsv[:,:,1]
    V = gbr_hsv[:,:,2]
    meanH = np.mean(H)
    WarnaH.append(meanH)
    meanS = np.mean(S)
    WarnaS.append(meanS)
    meanV = np.mean(V)
    WarnaV.append(meanV)
    
    keterangan = '1'
    Kondisi.append(keterangan)
    
data1 = pd.DataFrame (WarnaR, columns=['Warna R'])
data2 = pd.DataFrame (WarnaG, columns=['Warna G'])
data3 = pd.DataFrame (WarnaB, columns=['Warna B'])
data4 = pd.DataFrame (WarnaH, columns=['Warna H'])
data5 = pd.DataFrame (WarnaS, columns=['Warna S'])
data6 = pd.DataFrame (WarnaV, columns=['Warna V'])
data7 = pd.DataFrame (Kondisi, columns=['Kondisi'])

listdata2 = [data1, data2, data3, data4, data5, data6, data7]
gabung2 = pd.concat(listdata2, axis=1)


#export to Excel
datalist = [gabung, gabung2]
total = pd.concat(datalist, ignore_index=True)
total.to_csv("Extraksi Fitur Kolam Lele.csv", index=False)
total.to_excel("Extraksi Fitur Kolam Lele.xlsx", index=False)

print("Extrasi fitur berhasil")