import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import stats


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



def cleaning(data):
    gait = data.copy()

    gait['prev'] = gait['ay'].shift(periods=1)
    gait['next'] = gait['ay'].shift(periods=-1)
    gait['timeN'] = gait['time'].shift(periods=-1)
    gait = gait[['time','ay','next', 'timeN', 'prev']]
    gait.dropna(inplace=True)


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
    
    return result


def ttest(gait):
    # Can do transform
    normality_right = stats.normaltest(gait['right']).pvalue
    normality_left = stats.normaltest(gait['left']).pvalue

    ## Check for normal distribution and equal variance. If p > 0.05 then it's valid to do t-test
    test_variance = stats.levene(gait['right'], gait['left']).pvalue

    print("normality test right: ", normality_right)
    print("normality test left: ", normality_left)
    print("variance test: ", test_variance)

    ## Compute T-test. Null hypothese: right step and left step have the same step time. If p < 0.05, we reject the null hypothesis
    ttest = stats.ttest_ind(gait['right'], gait['left'])
    return ttest

def utest(gait):

    # Null hypothesis: left foot and right foot are equally likely the same in average time. 
    # If p < 0.05, we can reject this and say left != right. And there is imbalance
    utest =  stats.mannwhitneyu(gait['right'],gait['left'], alternative= "two-sided")
    return utest



if __name__ == '__main__':
    filename = sys.argv[1]
    #output = filename[0:-4] + 'result.csv'
    print(filename)
    data = pd.read_csv("data/" + filename)
    gait = filter(data)

    cleaned_data = cleaning(gait)

    ttest_result = ttest(cleaned_data)
    print(ttest_result)

    utest_result = utest(cleaned_data)
    print(utest_result)

    #plt.boxplot([cleaned_data['right'],cleaned_data['left']])
    #plt.show()
