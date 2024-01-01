import numpy as np 
import pandas as pd
mush=pd.read_csv("mushroom.csv")
mush.replace('?',np.nan,inplace=True)
print(len(mush.columns),"columns after dropping NA",len(mush.dropna(axis=1).columns))
mush.dropna(axis=1,inplace=True)
target='class'
features=mush.columns[mush.columns!=target]

classes=mush[target].unique()
test=mush.sample(frac=0.4)
mush=mush.drop(test.index)
probs={}
probcl={}
for x in classes:
    mushcl=mush[mush[target]==x][features]
    clsp={}
    tot=len(mushcl)
    for col in mushcl.columns:
        colp={}
        for val,cnt in mushcl[col].value_counts().items():
            pr=cnt/tot
            colp[val]=pr
            clsp[col]=colp
    probs[x]=clsp
    probcl[x]=len(mushcl)/len(mush)


def probas(x):
    if not isinstance(x,pd.Series):
        raise IOError('Arg must of type series')
    probab={}
    for cl in classes:
        pr=probcl[cl]
        for col,val in x.items():  
            try:
                pr*=probs[cl][col][val]
            except KeyError:
                pr=0
            probab[cl]=pr
        return probab

def classify(x):
    probab=probas(x)
    mx=0
    mxcl=0
    for cl,pr in probab.items():
        if pr>mx:
            mx=pr
            mxcl=cl
    return mxcl
b=[]
for i in mush.index:
    b.append(classify(mush.loc[i,features])==mush.loc[i,target])
print(sum(b),"correct of",len(mush))
print("Acuuracy:",sum(b)/len(mush))
b=[]
for i in test.index:
    b.append(classify(test.loc[i,features])==test.loc[i,target])
print(sum(b),"correct of",len(test))
print("Accuracy:",sum(b)/len(test))