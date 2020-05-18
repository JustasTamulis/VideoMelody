import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import psutil
import pickle
process = psutil.Process(os.getpid())

folder_name = "F:\\CompSci\\project\\Data\\Koyaanisqatsi\\"

ohe_path = folder_name + "OneHot\\hot2piano.npy"
rohe_path = folder_name + "OneHot\\piano2hot.pkl"

var = np.zeros((1,128),dtype=int)
for i in range(1,8):

    piano_path = folder_name + "midi_npy\\" + str(i) + '.npy'

    piano = np.load(piano_path)
    print("_________")
    print(i)
    print(piano.shape)
    uniq = np.unique(piano, axis = 0)
    print(uniq.shape)
    var = np.concatenate((var, uniq))
    print(var.shape)
    var = np.unique(var, axis = 0)
print("overall")
print(var.shape)

np.save(ohe_path, var)
dic  = dict()
for i in range(0,len(var)):
    strr = np.array2string(var[i,:], max_line_width = 100000)
    dic[strr] = i

f = open(rohe_path,"wb")
pickle.dump(dic,f)
f.close()
print("//saved//")

for i in range(1,8):

    piano_path = folder_name + "midi_npy\\" + str(i) + '.npy'
    ohe_path = folder_name + "OneHot\\" + str(i) + '.npy'

    piano = np.load(piano_path)
    piano = piano.astype(int)
    ohe = np.zeros(len(piano), dtype = int)
    for i in range(0, len(piano)):
        strtr = np.array2string(piano[i,:], max_line_width = 100000)
        ohe[i] = dic.get(strtr)

    print(ohe.shape)
    print(ohe[0])
    np.save(ohe_path, ohe)
