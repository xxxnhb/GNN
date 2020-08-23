from PIL import Image
import numpy as np
import csv
import pickle

data_=[]
with open('E:\picture\ISIC2018_Task3_Training_GroundTruth.csv','r') as csvfile:
    csv_reader = csv.reader(csvfile)
    type = next(csv_reader)
    for i in range(10015):
        data_.append(next(csv_reader))
type = np.array(type)
type = type[1:]
data_ = np.array(data_)
data_ = data_[:,1:]
data_ =data_.astype(float)
data_=data_.astype(int)
index=np.argmax(data_,1)


data0=[]
data1=[]
data2=[]
data3=[]

data5=[]
data6=[]

for i in range(10015):
     path = 'E:\picture\ISIC2018_Task3_Training_Input\ISIC_00' + str(i + 24306) + '.jpg'
     img = Image.open(path)
     img = img.resize((84,84), Image.ANTIALIAS)
     img = np.array(img)
     if index[i] == 0:
        data0.append(img)
     elif index[i] == 1:
        data1.append(img)
     elif index[i] == 2:
        data2.append(img)
     elif index[i] == 3:
         data3.append(img)
     elif index[i] == 5:
         data5.append(img)
     elif index[i] == 6:
         data6.append(img)
data0_train=data0[:723]
data0_val=data0[723:890]
data0_test=data0[890:]
data1_train=data1[:4358]
data1_val=data1[4358:5364]
data1_test=data1[5364:]
data2_train=data2[:334]
data2_val=data2[334:411]
data2_test=data2[411:]
data3_train=data3[:212]
data3_val=data3[212:262]
data3_test=data3[262:]

data5_train=data5[:75]
data5_val=data5[75:92]
data5_test=data5[92:]
data6_train=data6[:92]
data6_val=data6[92:114]
data6_test=data6[114:]

dataA=[]
for i in range(365):
    path='E:/dataset/image/A/image_A ('+str(i+1)+').jpg'
    img = Image.open(path)
    img = img.resize((84,84), Image.ANTIALIAS)
    img = np.array(img)
    dataA.append(img)

dataB=[]
for i in range(602):
    path='E:/dataset/image/B/image_B ('+str(i+1)+').jpg'
    img = Image.open(path)
    img = img.resize((84,84), Image.ANTIALIAS)
    img = np.array(img)
    dataB.append(img)

dataC=[]
for i in range(2610):
    path = 'E:/dataset/image/C/image_C (' + str(i + 1) + ').jpg'
    img = Image.open(path)
    img = img.resize((84, 84), Image.ANTIALIAS)
    img = np.array(img)
    dataC.append(img)

dataD=[]
for i in range(205):
    path='E:/dataset/image/D/image_D ('+str(i+1)+').jpg'
    img = Image.open(path)
    img = img.resize((84,84), Image.ANTIALIAS)
    img = np.array(img)
    dataD.append(img)

dataE=[]
for i in range(1021):
    path='E:/dataset/image/E/image_E ('+str(i+1)+').jpg'
    img = Image.open(path)
    img = img.resize((84,84), Image.ANTIALIAS)
    img = np.array(img)
    dataE.append(img)

dataG=[]
for i in range(377):
    path='E:/dataset/image/G/image_G ('+str(i+1)+').jpg'
    img = Image.open(path)
    img = img.resize((84,84), Image.ANTIALIAS)
    img = np.array(img)
    dataG.append(img)

dataH=[]
for i in range(391):
    path='E:/dataset/image/H/image_H ('+str(i+1)+').jpg'
    img = Image.open(path)
    img = img.resize((84,84), Image.ANTIALIAS)
    img = np.array(img)
    dataH.append(img)

dataA_train=dataA[:238]
dataA_val=dataA[238:292]
dataA_test=dataA[292:]
dataB_train=dataB[:391]
dataB_val=dataB[391:481]
dataB_test=dataB[481:]
dataC_train=dataC[:1697]
dataC_val=dataC[1697:2088]
dataC_test=dataC[2088:]
dataD_train=dataD[:134]
dataD_val=dataD[134:164]
dataD_test=dataD[164:]
dataE_train=dataE[:664]
dataE_val=dataE[664:816]
dataE_test=dataE[816:]
dataG_train=dataG[:245]
dataG_val=dataG[245:301]
dataG_test=dataG[301:]
dataH_train=dataH[:255]
dataH_val=dataH[255:312]
dataH_test=dataH[312:]

data_train_84={0:data0_train,1:data1_train,2:data2_train,3:data3_train,4:data5_train,5:data6_train,
          6:dataA_train,7:dataB_train,8:dataC_train,9:dataD_train,10:dataE_train,11:dataG_train,12:dataH_train
          }
data_val_84={0:data0_val,1:data1_val,2:data2_val,3:data3_val,4:data5_val,5:data6_val,
          6:dataA_val,7:dataB_val,8:dataC_val,9:dataD_val,10:dataE_val,11:dataG_val,12:dataH_val
          }
data_test_84={0:data0_test,1:data1_test,2:data2_test,3:data3_test,4:data5_test,5:data6_test,
          6:dataA_test,7:dataB_test,8:dataC_test,9:dataD_test,10:dataE_test,11:dataG_test,12:dataH_test
          }

with open('mini_imagenet_train.pickle', 'wb') as f:
    pickle.dump(data_train_84, f)
with open('mini_imagenet_val.pickle', 'wb') as f:
    pickle.dump(data_val_84, f)
with open('mini_imagenet_test.pickle', 'wb') as f:
    pickle.dump(data_test_84, f)