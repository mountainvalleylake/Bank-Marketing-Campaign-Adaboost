import pandas as pd
import math
import random
def changeDataset():
    dataframe1 = pd.read_csv('D:\Study\Python Codes\MLOffline1\Data\\bank-additional-yes.csv',encoding='utf-8')
    dataframe2 = pd.read_csv('D:\Study\Python Codes\MLOffline1\Data\\bank-additional-no.csv',encoding='utf-8')
    col_list = dataframe1.columns.values
    dataframe = pd.DataFrame(columns=col_list)
    l1 = len(dataframe1)
    l2 = len(dataframe2)
    print(l1,l2)
    for i in range(0,l1+l2):
        idx = math.floor(i / 2)
        if i%2==0:
            element = dataframe1.iloc[idx]
            dataframe.loc[len(dataframe)] = element
        else:
            element = dataframe2.iloc[idx]
            dataframe.loc[len(dataframe)] = element
    dataframe.to_csv('D:\Study\Python Codes\MLOffline1\Data\\bank-additional.csv')
    print(len(dataframe))
def arrangeDataset():
    dataframe = pd.read_csv('D:\Study\Python Codes\MLOffline1\Data\\bank-additional-full.csv',encoding='utf-8')
    col_list = dataframe.columns.values
    ydf = dataframe.loc[dataframe['y'] == 'yes']
    ylist = ydf.values.tolist()
    ndf = dataframe.loc[dataframe['y'] == 'no']
    nlist = ndf.values.tolist()
    if len(ylist) < len(nlist):
        margin = len(ylist)
    else:
        margin = len(nlist)
    target = margin
    simil = []
    simil.append(-1)
    nfin = []
    r = 0
    while(target > 0):
        #r = random.randint(0,len(nlist)-1)
        #if r in simil:
        #    continue
        #else:
        val = nlist[r]
        #simil.append(r)
        nfin.append(val)
        target -= 1
        r += 1
    newdataframe = pd.DataFrame(columns=col_list)
    l1 = len(ylist)
    l2 = len(nfin)
    for i in range(0,l1+l2):
        idx = math.floor(i / 2)
        if i%2==0:
            element = ylist[idx]
            newdataframe.loc[len(newdataframe)] = element
        else:
            element = nfin[idx]
            newdataframe.loc[len(newdataframe)] = element
    newdataframe.to_csv('D:\Study\Python Codes\MLOffline1\Data\\bank-additional-full-new.csv')
    print(len(newdataframe))
    #print(col_list)
def main():
    #changeDataset()
    arrangeDataset()
if __name__ == '__main__':
    main()