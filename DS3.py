import glob
import os

def read(path="\datasets\\3"):
    X = []
    Y = []
    control = [];
    parkinson = [];
    h_parkinson = []

    for filename in os.listdir(os.getcwd()+path+"\hw_dataset\control"):
        if filename.endswith(".txt"):
            f = open(os.getcwd()+path+"\hw_dataset\control\\"+filename, 'r')
            control.append(f.readlines());
    for filename in os.listdir(os.getcwd()+path+"\hw_dataset\parkinson"):
        if filename.endswith(".txt"):
            f = open(os.getcwd()+path+"\hw_dataset\parkinson\\"+filename, 'r')
            parkinson.append(f.readlines());

    for filename in os.listdir(os.getcwd()+path+"\\new_dataset\parkinson"):
        if filename.endswith(".txt"):
            f = open(os.getcwd()+path+"\\new_dataset\parkinson\\"+filename, 'r')
            h_parkinson.append(f.readlines());



def train():
    read()
