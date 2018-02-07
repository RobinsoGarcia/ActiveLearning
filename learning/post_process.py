import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
#%matplotlib inline
#plt.style.use('ggplot')

def get_charts(datafolder,sim):
    sim = str(sim)
    hmd = datafolder+'/'
    random = np.load(hmd+'random'+sim+'.npy')
    random_mean = np.mean(random,axis=0)
    random_std = np.std(random,axis=0)
    array = np.load(hmd+'results'+sim+'.npy').tolist()
    scenario_per={}
    for j in array.keys():
        bin_ = array[j]
        case_per={}
        for k in array[j].keys():
            if 'random.' in k:
                continue
            mean = array[j][k]-random_mean
            upl = array[j][k]-(random_mean+random_std)
            lol = array[j][k] - (random_mean-random_std)
            case_per[k] = np.sum(mean)
            #print('case'+j+'-'+k)
            #print('random',random_mean)
            #print('active', array[j][k])
            #input("Press Enter to continue...")

            plt.figure()
            plt.title(j+': '+k)
            plt.ylabel('accuracy gain')
            plt.xlabel('number of labelled samples')
            plt.fill_between(np.arange(mean.shape[0]),upl,lol,alpha=0.1)
            plt.plot(np.arange(mean.shape[0]),mean)
            plt.savefig(hmd+j+'/'+k+sim)
            scenario_per[k]=np.sum(mean)

            plt.figure()
            plt.title(j+': '+k)
            plt.ylabel('accuracy')
            plt.xlabel('number of labelled samples')

            plt.plot(np.arange(mean.shape[0]),array[j][k],label='activeL')
            plt.plot(np.arange(mean.shape[0]),random_mean,label='random')
            plt.legend()
            plt.savefig(hmd+j+'/'+k+'_accuracy'+sim)


    np.save(hmd+'indicators'+sim+'.npy',scenario_per)
    print(scenario_per)
    pass


def build_dataframe(datafolder,sim):
    data=np.load(datafolder+'/indicators'+sim+'.npy').tolist()
    dframe = pd.DataFrame.from_dict(data,orient='index')
    dframe.columns = [datafolder]
    return dframe

def build_table(folder_list):
    dframe=build_dataframe(folder_list[0])
    for i in folder_list[1:]:
        dframe = pd.concat([dframe,build_dataframe(i)],axis=1,join='inner')
