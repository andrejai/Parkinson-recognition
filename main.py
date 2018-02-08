import DS2_read as DS2
import os

if __name__ == "__main__":
    print(os.getcwd())
    DS2.read(os.getcwd()+"\datasets\\2\parkinsons_updrs.data.csv")
