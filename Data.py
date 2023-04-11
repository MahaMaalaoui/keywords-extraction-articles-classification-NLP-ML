
import pandas as pd 
import numpy as np




df0 = pd.read_csv("/data/covid_abstracts.csv")
df0=df0['title']+df0['abstract']


df1 = pd.read_csv("/data/train.csv")
df1=df1[:10000]
df1=df1['TITLE']+df1['ABSTRACT']

data = pd.concat([df0, df1], axis=0)


from numpy import  ones,zeros
Y0=zeros((10000, )) 
Y1=ones((10000, )) 

labels =np.concatenate((Y0, Y1), axis=0)

data1=pd.DataFrame()
data1['text']=data
data1['labels']=labels


from sklearn.utils import shuffle
df= shuffle(data1)


def getdata():
    
    return(df)


if __name__ == '__main__':
  getdata()
 
