import random
import numpy as np
import pandas as pd
import math

dataframe = None
numlist,nonnumlist = None, None #contains name only
t,total_pos,total_neg = 0,0,0 #total size, pos and neg of resampled dataset
attribute_info_list = []
datapoints = None
classifier_list = []
weight_vector = []
z = []
fscore = []
total_folds = 0
fold_data = []
train_data = None
test_data = None
class AttributeInfo: #split value,p,ng,ip,ing all are lists
    def __init__(self,status,attribute,split_value,gain,p,ng):
        self.status = status
        self.attribute = attribute
        self.split_value = split_value
        self.gain = gain
        self.p = p
        self.ng = ng
        self.verdict = []
        #decide leftside rightside here
        if self.status == 'Non numeric':
            for i in range(0,len(self.split_value)):
                if self.p[i] > self.ng[i]:
                    self.verdict.append('yes')
                else:
                    self.verdict.append('no')
        else:
            for i in range(0,2):
                if self.p[i] > self.ng[i]:
                    self.verdict.append('yes')
                else:
                    self.verdict.append('no')


    def printClass(self):
        print(self.status)
        print(self.attribute)
        print(self.split_value)
        print(self.gain)
        print(self.p)
        print(self.ng)
        print(self.verdict)

    def retGain(self):
        return self.gain

    def retSplit(self):
        return self.split_value

    def retPos(self):
        return self.p

    def retNeg(self):
        return self.ng

    def retAttr(self):
        return self.attribute

    def retStatus(self):
        return self.status

    def retVerdict(self):
        return self.verdict


def readData(path):
    global dataframe,datapoints,numlist,nonnumlist
    dataframe = pd.read_csv(path,encoding='utf-8')
    datapoints = dataframe.values.tolist()
    #print(len(datapoints))
    #Add the numerical types attribute names into a list 'numlist'
    numlist = list(dataframe.select_dtypes(include=['int64']).columns)
    flist = list(dataframe.select_dtypes(include=['float64']).columns)
    for f in flist:
        numlist.append(f)
    # Add the non-numerical types attribute names into a list 'nonnumlist'
    nonnumlist = list(dataframe.select_dtypes(include=['O']).columns)
    nonnumlist.remove('y')


def Normalize(w):
    sum = 0
    for i in range(0,len(w)):
        sum += w[i]
    for i in range(0,len(w)):
        w[i] = w[i]/sum
    return w


def calcEntropy(pos, neg):
    if pos == 0 or neg == 0:
        return 0
    #print(pos,neg)
    t = pos + neg
    #print(t)
    pos_ratio = pos / t
    #print(pos_ratio)
    entropy = -(pos_ratio * math.log(pos_ratio,2) + (1-pos_ratio) * math.log((1-pos_ratio),2))
    return entropy


def calcInformationGain(pos,neg,total):
    ig = ((pos+neg)/total) * calcEntropy(pos,neg)
    return ig


def initCalculation(df):
    global dataframe, numlist, nonnumlist, attribute_info_list
    global t, total_pos, total_neg
    total = df.groupby(['y'])
    total_pos, total_neg = 0, 0
    for nm, gp in total:
        # print(nm)
        if nm == 'yes':
            total_pos = gp['y'].count()
        else:
            total_neg = gp['y'].count()
    # print(total_pos,total_neg)
    t = total_pos + total_neg


def splitNonNumeric(df):
    global dataframe, numlist, nonnumlist, attribute_info_list
    global t, total_pos, total_neg
    Initial_Entropy = calcEntropy(total_pos, total_neg)
    #print(Initial_Entropy)
    alist = []
    for n in nonnumlist:
        vlist = df[n].values
        atrlist = list(set(vlist))
        alist.append(atrlist)
    i = -1
    for n in nonnumlist:
        i += 1
        alistx = alist[i]
        grped = df.groupby([n, 'y'])
        entropy = 0
        pos, neg = 0, 0
        plist, nlist,atrl = [],[],[]
        for a in alistx:
            for name,grp in grped:
                if name[0] == a:
                    if name[1] == 'no':
                        neg += grp['y'].count()
                        pos += 0
                        # print(neg,'no')
                    elif name[1] == 'yes':
                        neg += 0
                        pos += grp['y'].count()
                else:
                    continue
            plist.append(pos)
            nlist.append(neg)
            atrl.append(a)
            entropy += calcInformationGain(pos,neg,t)
            pos, neg = 0, 0
        ig = Initial_Entropy - entropy
        x = AttributeInfo('Non numeric', n, atrl, ig, plist, nlist)
        attribute_info_list.append(x)


def splitNumeric(df):
    global dataframe, numlist, nonnumlist, attribute_info_list
    global t, total_pos, total_neg
    Initial_Entropy = calcEntropy(total_pos, total_neg)
    maxtropy = -9999
    maxattrval = -9999
    pos, neg = 0, 0
    p, ng, ip, ing = 0, 0, 0, 0
    for n in numlist:
        # print(n)
        # sort the entire dataset by one column 'n'
        res = df.sort_values(n)
        # print(res.head(10))
        # get the number of yes and no for every unique element of a column
        grped = res.groupby([n, 'y'])
        for name, grp in grped:
            # print("points ", name[0])
            if name[1] == 'no':
                neg += grp['y'].count()
                pos += 0
                # print(neg,'no')
            elif name[1] == 'yes':
                neg += 0
                pos += grp['y'].count()
                # print(pos,'yes')
            ipos = total_pos - pos
            ineg = total_neg - neg
            if ipos == 0 and ineg == 0:
                break
            ig = Initial_Entropy - calcInformationGain(pos, neg, t) - calcInformationGain(ipos, ineg, t)
            # print(ig)
            if ig > maxtropy:
                maxtropy = ig
                maxattrval = name[0]
                p = pos
                ng = neg
                ip = ipos
                ing = ineg
                # print(maxattrval,maxtropy)
        alist,plist,nglist = [],[],[]
        alist.append(maxattrval)
        plist.append(p)
        plist.append(ip)
        nglist.append(ng)
        nglist.append(ing)
        a = AttributeInfo('Numeric',n, alist, maxtropy, plist, nglist)
        attribute_info_list.append(a)
        #print(n, maxattrval, maxtropy)

def kfolding(k):
    global total_folds
    total_folds = k
    l = len(datapoints)
    efold = math.floor(l / k)
    for i in range(0,k):
        #print(i)
        lst = datapoints[i*efold:(i+1)*efold]
        #print(lst)
        fold_data.append(lst)

def traintestSep(k):
    global total_folds,train_data,test_data
    tlist = fold_data[k]
    rlist = []
    for i in range(0,total_folds):
        if i == k:
            continue
        else:
            temp = fold_data[i]
            rlist.extend(temp)
    col_list = dataframe.columns.values
    train_data = pd.DataFrame.from_records(rlist,columns=col_list)
    test_data = pd.DataFrame.from_records(tlist,columns=col_list)


def selectBestAttribute(training_df):
    global attribute_info_list
    initCalculation(training_df)  #send the resampled dataframe
    splitNumeric(training_df)  #send the resampled dataframe
    splitNonNumeric(training_df)  #send the resampled dataframe
    gain = -1
    cls = None
    for a in attribute_info_list:
        #a.printClass()
        val = a.retGain()
        if val > gain:
            gain = val
            cls = a
    attribute_info_list.clear()  # empty this helper list
    return cls


def weightedResample(w):
    #shuffle datapoints
    global datapoints
    np.random.shuffle(datapoints)
    target = len(w) -1
    col_list = train_data.columns.values
    df = pd.DataFrame(columns=col_list)
    cumi = []
    s,idx = 0,0
    for i in range (0,len(w)):
        s += w[i]
        cumi.append(s)
    while(target >= 0):
        r1 = random.uniform(0,1) #select an index
        #print(r1)
        for j in range(0,len(cumi)):
            if r1 <= cumi[j]:
                idx = j
                break
            #print('Yahoo')
        element = datapoints[idx]
        df.loc[len(df)] = element
        #print(element)
        target -= 1
    #df = pd.DataFrame.append(element)
    #print(df)
    return df


def Adaboost():
    global weight_vector,dataframe,z,classifier_list
    global datapoints,train_data,test_data
    path = 'D:\Study\Python Codes\MLOffline1\Data\\bank-additional-full-new.csv'
    readData(path)
    nfl = 30
    ncl = 5
    kfolding(nfl)  # how many folds
    error = 0
    for fold in range(0, nfl):
        print("Fold ",fold)
        traintestSep(fold)
        datapoints = train_data.values.tolist()
        #now boosting starts from here
        for i in range(0, len(datapoints)):
            weight_vector.append(1/len(datapoints))
        #print(weight_vector)
        verdict = train_data['y'].values
        #the first learner L0
        cla_name_list = []
        for k in range(0, ncl): #make 5  classifiers
            #resample here on the basis of w
            training_df = weightedResample(weight_vector)
            attribute_sel = selectBestAttribute(training_df) #this is the attribute hypothesis
            #attribute_sel.printClass()
            sel = attribute_sel.retAttr()
            kind = attribute_sel.retStatus()
            feature = train_data[sel].values
            split = attribute_sel.retSplit()
            #print(split)
            vdct = attribute_sel.retVerdict()
            res = []
            for j in range(0,len(datapoints)):
                if kind == 'Numeric':
                    if feature[j] >= split[0]:
                        r = vdct[0]
                    else:
                        r = vdct[1]
                else:
                    try:
                        m = split.index(feature[j])
                        r = vdct[m]
                    except ValueError:
                        m = -1
                        if total_pos > total_neg:
                            r = 'yes'
                        else:
                            r = 'no'
                res.append(r)
                if r != verdict[j]:
                     error += weight_vector[j]
            for j in range(0, len(datapoints)):
                if res[j] == verdict[j]:
                    weight_vector[j] = weight_vector[j] * error /(1-error)
            weight_vector = Normalize(weight_vector)
            z.append(math.log((1-error)/error,2))
            classifier_list.append(attribute_sel)
            cla_name_list.append(sel)
        #sumval = sum(z)
        print(error)
        #print("weighted sum", sumval)
        #test on the remaining fold
    if error < 0.5:
        pp, nn, pn, np = 0, 0, 0, 0
        # cidx = []
        # cidx.append(-1)
        test_datapoints = test_data.values.tolist()
        for d in range(0,len(test_datapoints)):
            minidf = test_data.loc[d]
            #print(minidf)
            fin = minidf['y']
            tres = []
            for k in range(0,ncl):
                tattribute_sel = classifier_list[k]
                tsel = tattribute_sel.retAttr()
                x = minidf[tsel]
                #print(tsel)
                kind = tattribute_sel.retStatus()
                split = tattribute_sel.retSplit()
                # print(split)
                tvdct = tattribute_sel.retVerdict()
                if kind == 'Numeric':
                    if x >= split[0]:
                        r = tvdct[0]
                    else:
                        r = tvdct[1]
                else:
                    try:
                        m = split.index(x)
                        r = tvdct[m]
                    except ValueError:
                        m = -1
                        if total_pos > total_neg:
                            r = 'yes'
                        else:
                            r = 'no'
                tres.append(r)
            yw,nw = 0,0
            for k in range(0,ncl):
                if tres[k] == 'yes':
                    yw += z[k]
                else:
                    nw += z[k]
            if yw > nw:
                v = 'yes'
            else:
                v = 'no'
            if v == fin:
                if v == 'yes':
                    pp += 1
                else:
                    nn += 1
            else:
                if fin == 'yes':
                    pn += 1
                else:
                    np += 1
        #print(pp,nn,pn,np)
        try:
            recall = pp / (pp + pn)
            precision = pp / (pp + np)
            fs = 2/((1/recall)+(1/precision))
            print("Fscore ", fs)
            fscore.append(fs)
        except ZeroDivisionError as err:
            print('Handling run-time error:', err)
            print('Fscore nan')
            fscore.append('nan')
        #print(classifier_list)
        #print(z)
        z.clear()
        classifier_list.clear()
        weight_vector.clear()
        train_data = None
        test_data = None
        datapoints.clear()


def main():
    Adaboost()

if __name__ == '__main__':
    main()

#
#total_rows = dataframe.shape[0] #get total number of rows
#print(total_rows)
#
# col = dataframe.columns.values #get column names into a list
# print(col)
#
# Grouped Columns According to Data type
# g = dataframe.columns.to_series().groupby(dataframe.dtypes).groups
# print(g)
#
# u = list(dataframe.iloc[3]) #save a row into a list by location
# print(u)
# v = list(dataframe[numlist[1]].values) #access a numeric column
# print(v)
#
#df = df.iloc[0:0]  # empty a dataframe of all datas