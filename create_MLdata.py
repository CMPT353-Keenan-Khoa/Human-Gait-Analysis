import sys
import pandas as pd
import numpy as np
from scipy import signal
#import plotly.express as px


def filter(gait):
    data = gait.copy()

    columns = data.columns

    if 'ay (m/s^2)' in columns:
        data = data.rename(columns={"ay (m/s^2)": "ay"})

    #time decision
    leng = len(gait['time'])
    cut = int(leng*0.1)
    data = data.loc[(gait['time']>gait['time'].values[cut])&(gait['time']<gait['time'].values[leng-cut])]


    #butter filter
    b, a = signal.butter(3, 0.05, btype='lowpass', analog=False)
    data['ay'] = signal.filtfilt(b, a, data['ay'])
    return data

def step(data):
    gait = data.copy()

    gaitleft = gait.loc[((gait['ay']<0)&(gait['next']>0))]
    gaitright =  gait.loc[((gait['ay']>0)&(gait['next']<0))]
    gaitleft = gaitleft.reset_index()
    gaitright = gaitright.reset_index()


    gaitright['timeN'] = gaitright['time'].shift(periods=-1)
    gaitleft['timeN'] = gaitleft['time'].shift(periods=-1)


    if gaitleft['time'][0] > gaitright['time'][0]:
        steptimeR = gaitleft['time'] - gaitright['time']
        steptimeL = gaitright['timeN'] - gaitleft['time']
    else:
        steptimeL = gaitright['time'] - gaitleft['time']
        steptimeR = gaitleft['timeN'] - gaitright['time']


    steptimeR.dropna(inplace=True)
    steptimeL.dropna(inplace=True)

    result = pd.concat([steptimeR,steptimeL], axis=1, sort=False, ignore_index=False)
    result = result.rename(columns={"time": "right", 0: "left"})
    result = result.loc[(result['right']>0.3)&(result['right']<1.5)]
    result = result.loc[(result['left']>0.3)&(result['left']<1.5)]
    step = result['right'].count()
    
    return step

def distance(data):
    gait = data.copy()

    gait = gait.loc[gait['error']==False]
    
    gait['speed'] = gait['ay'] * (gait['timeN']-gait['time'])
    gait['speedP'] = gait['speed'].shift(periods=1)
    gait.dropna(inplace=True)
    gait['distance(cm)'] = gait['speed']**2 - gait['speedP']**2 / (gait['ay']*2)
    gait['distance(cm)'] = gait['distance(cm)'] * 100
    distance = gait['distance(cm)'].values.sum()

    return distance

def time(data):
    gait = data.copy()

    gaitgap = gait.loc[gait['gap']>1]
    gaitgap = gaitgap.sum()
    remove = gaitgap['gap']
    
    time = (gait['time'].values[len(gait['time'])-1] - gait['time'].values[0]) - remove

    return time

def calculation(data,split):
    gait = data.copy()
    leng = len(gait['time'])
    cut = int(leng*(1/split))
    resultS = np.zeros(split)
    resultT = np.zeros(split)
    resultD = np.zeros(split)
    for i in range (split):
        splited = gait.loc[(gait['time']>gait['time'].values[cut*(i)])&(gait['time']<gait['time'].values[cut*(i+1)-1])]
        resultS[i] = step(splited)
        resultT[i] = time(splited)
        resultD[i] = distance(splited)
    #print(resultS)
    #print(resultD)
    #print(resultT)

    #print(resultD.sum())
    #print(resultS.sum())
    #print(resultT.sum())
    return resultS, resultT, resultD
        

def cleaning(data):
    gait = data.copy()
    leng = len(gait['time'])
    cut = int(leng*0.1)

    gait['prev'] = gait['ay'].shift(periods=1)
    gait['next'] = gait['ay'].shift(periods=-1)
    gait['timeN'] = gait['time'].shift(periods=-1)
    gait['gap'] = gait['timeN']-gait['time']
    gait = gait[['time','ay','next', 'timeN', 'prev', 'gap']]
    gait.dropna(inplace=True)
    gait['error']= gait['gap']>1
    gaitgap = gait.loc[gait['gap']>1]
    gaitgap = gaitgap.sum()
    remove = gaitgap['gap']
    

    #total time taken calculation
    timetaken = (gait['time'].values[len(gait['time'])-1] - gait['time'].values[0]) - remove

    if timetaken < 60:
        split=2
    elif timetaken < 120:
        split=4
    elif timetaken < 180:
        split=6
    elif timetaken < 240:
        split=8
    elif timetaken < 300:
        split=10
    elif timetaken < 360:
        split=12
    elif timetaken < 420:
        split=14
    elif timetaken < 480:
        split=16
    elif timetaken < 540:
        split=18
    elif timetaken >= 540:
        split=20
    
    result = calculation(gait,split)
    return result


def main(filename, height):
    # filename = 'keenan.csv'
    #output_filename = 'MLdata.csv'
    # output = filename[0:-4] + 'Result.csv'
    # height = 168
    data = pd.read_csv("data/" + filename)
    #output = pd.read_csv(output_filename)

    gait = filter(data)
    step, time, distance = cleaning(gait)

    final = {'step':step,
             'distance':distance,
             'time':time}

    result = pd.DataFrame(final,columns=['step','distance','time'])
    result['height'] = height
    return result

def classify_height(h):
    group = 0 
    if h <= 150:
        group = 0
    elif h <= 160:
        group = 1
    elif h <= 170:
        group = 2
    elif h <= 180:
        group = 3
    elif h <= 190:
        group = 4
    return group
    
    
if __name__ == "__main__":
    khoa = main('khoa.csv', 171)
    khoa2 = main('khoa2.csv', 170)
    keenan = main('keenan.csv', 168)
    sun = main('sun.csv', 153)
    keenan2 = main('keenan2.csv', 168)
    yeon = main('yeon.csv', 163)
    pt = main ('pt.csv', 169)
    phong = main('phong.csv', 165)
    sang = main('sang.csv', 187)
    jin = main('jin.csv', 185)
    fake = main('fake.csv', 168)
    
    result = pd.concat([khoa, khoa2, keenan, pt, phong, sang, jin,keenan2, sun, yeon], axis=0, sort=False)
    result['pace'] = result['step']/result['time']
    result['step_length'] = result['distance']/result['step']

    result['range'] = result['height'].map(classify_height)
    count = result.copy()
    count['count'] = 1
    count = count.groupby('range').sum()
    #print(count)
    #print(result)

    graph = result[['step', 'pace', 'height']]
    #fig = px.scatter_3d(graph, x='step', y='pace', z='height')
    #fig.show()
    result.to_csv('mldata.csv', index=False)
