import numpy as np
import pandas as pd
eps=np.finfo(float).eps
from numpy import log2 as log
outlook='overcast,overcast,overcast,overcast,overcast,rainy,rainy,rainy,rainy,sunny,sunny,sunny,sunny,sunny'.split(',')
temp='hot,cool,mild,hot,mild,cool,cool,mild,mild,hot,hot,mild,cool,mild'.split(',')
humidity='high,normal,high,normal,high,normal,normal,normal,high,high,high,high,normal,normal'.split(',')
windy='FALSE,TRUE,TRUE,FAlSE,FALSE,FALSE,TRUE,FALSE,TRUE,FALSE,TRUE,FALSE,FALSE,TRUE'.split(',')
play='yes,yes,yes,yes,yes,yes,no,yes,no,no,no,no,yes,yes'.split(',')

dataset={'outlook':outlook,'temp':temp,'humidity':humidity,'windy':windy,'play':play}
df=pd.DataFrame(dataset,columns=['outlook','temp','humidity','windy','play'])
print(df)
def find_entropy(df):
    Class=df.keys()[-1]
    entropy=0
    values=df[Class].unique()
    for value in values:
        fraction=df[Class].value_counts()[value]/len(df[Class])
        entropy+=-fraction*np.log2(fraction)
    return entropy

def find_entropy_attribute(df,attribute):
    Class=df.keys()[-1]
    target_variables=df[Class].unique()
    variables=df[attribute].unique()
    entropy2=0
    for variable in variables:
        entropy=0
        for target_variable in target_variables:
            num=len(df[attribute][df[attribute]==variable][df[Class]==target_variable])
            den=len(df[attribute][df[attribute]==variable])
            fraction=num/(den+eps)
            entropy+=-fraction*log(fraction+eps)
        fraction2=den/len(df)
        entropy2+=-fraction2*entropy
    return abs(entropy2)
def get_subtable(df,node,value):
    return df[df[node]==value].reset_index(drop=True)
def find_winner(df):
    IG=[]
    for key in df.keys()[:-1]:
        IG.append(find_entropy(df)-find_entropy_attribute(df,key))
    return df.keys()[:-1][np.argmax(IG)]
def buildtree(df,tree=None):
    Class=df.keys()[:-1]
    node=find_winner(df)
    attvalue=np.unique(df[node])
    if tree is None:
        tree={}
        tree[node]={}
    for value in attvalue:
        subtable=get_subtable(df,node,value)
        clvalue,counts=np.unique(subtable['play'],return_counts=True)
        if len(counts)==1:
            tree[node][value]=clvalue[0]
        else:
            tree[node][value]=buildtree(subtable)
    return tree
t=buildtree(df)
import pprint
pprint.pprint(t)