import pandas as pd
import re

import os

os.chdir('G:/academics/ML2 Project/drug_discovery')

def get_target(c):
    target=[]
    pattern_t=re.compile(r'\d\s"')
    for item in c:
        t=re.findall(pattern_t,item)
        target.append(int(re.findall(re.compile('\d'),t[0])[0]))
    return target

def get_fingerprint(c):
    fingerprint=[]
    pattern_f=re.compile(r'"\d+"')
    for item in c:
        t=re.findall(pattern_f,item)
        t=re.findall(re.compile('\d+'),t[0])
        t=[int(d) for d in str(t[0])]
        fingerprint.append(t)
    return fingerprint

def occp(data):
    occp=[]
    for i in range(len(data.iloc[:,0])):
        occp_1=int(data.iloc[i,0][-1])
        occp_1024=int(data.iloc[i,-1][0])
        o=[occp_1]
        for t in range(1,data.shape[1]-1):
            o.append(data.iloc[i,t])
        o.append(occp_1024)
        occp.append(o)
        print(i,o)
    return occp

def get_type(c):
    type=[]
    pattern_type=re.compile(r'"\D\w+"')
    for item in c:
        t=re.findall(pattern_type,item)
        type.append(re.findall(re.compile('\w+'),t[0])[0])
    return type

def get_value(c):
    value=[]
    pattern_v=re.compile(r'"\d\S+"')
    for item in c:
        t=re.findall(pattern_v,item)
        value.append(float(re.findall(re.compile(r'\d\S+\d'),t[0])[0]))
    return value


data=pd.read_csv('myFP_217_D2.csv',header=None)
target=get_target(data.iloc[:,0])
print(len(target))
fingerprint=get_fingerprint(data.iloc[:,0])
print(len(fingerprint))
measure_type=get_type(data.iloc[:,-1])
print(len(measure_type))
measure_value=get_value(data.iloc[:,-1])
print(len(measure_value))
occp_v=occp(data)
print(len(occp_v))
#print(occp(data))
dataset=[]
for s in range(data.shape[0]):
    r=[]
    r=r+fingerprint[s]
    r=r+occp_v[s]
    r.append(measure_value[s])
    r.append(measure_type[s])
    r.append(target[s])
    dataset.append(r)


dataset=pd.DataFrame(dataset)
dataset.to_csv('processed.csv')