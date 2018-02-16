from learning.active import active as act
import learning.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from learning.post_process import get_charts,build_dataframe
import time

sims = np.arange(3)+8+30
sets = ['crx','breast_cancer','wine','tictac','pama-indians','liver','iris','digits']

for w in sims:
    for set_ in sets:
        print(set_)
        start_set=time.time()
        X,y,X_,y_=datasets.load(set_)
        sim=w
        min_pool = round(X.shape[0]*0.05)
        sample_size = round((X.shape[0]-min_pool)*0.7/20)
        iter_=20

        scenario_data = {'uncertainty':{0:['least_confident','none','none'],
        1:['margin','none','none'],
        2:['entropy','none','none'],
        3:['least_confident','euclidean','none'],
        4:['margin','euclidean','none'],
        5:['entropy','euclidean','none']},
        'boosting':{0:['boosting','euclidean','margin']},
        'bagging':{0:['bagging','euclidean','margin']}}

        data_folder = set_
        n_random=10

        numb_scenarios = len(scenario_data)
        results={}

        for i in scenario_data.keys():
            print('scenario {}'.format(i))
            num_cases = len(scenario_data[i])
            result={}
            for j in np.arange(num_cases):
                print("case {}".format(scenario_data[i][j]))
                case = scenario_data[i][j]
                obj = act(X,y,X_,y_,method=case[0],density_method=case[1])
                obj.qbc_sampling=0.4
                obj.n_splits=1000
                obj.min_pool = min_pool
                obj.sample_size = sample_size
                obj.set_pool()
                obj.set_density()
                obj.iter=iter_
                obj.qbc_method = case[2]
                obj.fit()
                result[case[0]+'-'+case[1]+'-'+case[2]]=obj.accuracy_bin
                #np.save('main/'+data_folder+'/'+i+'/'+case[0]+'-'+case[1]+'-'+case[2]+'.npy',result)
            results[i]=result

        print(results)
        #np.save('main/'+data_folder+'/'+'results'+str(sim)+'.npy',results)
        print('getting random')
e
        rand=[]
        for j in np.arange(n_random):
            rand.append([])
            obj2 = act(X,y,X_,y_,method='random')
            obj2.min_pool = min_pool
            obj2.sample_size = sample_size
            obj2.U_.shape
            obj2.set_pool()
            obj2.set_density()
            obj2.iter=iter_
            obj2.fit()
            rand[j]=obj2.accuracy_bin

        #np.save('main/'+ data_folder + '/random'+str(sim)+'.npy',rand)
        #print('get charts')
        #get_charts('main/'+data_folder,sim)

        #dframe = build_dataframe('main/'+data_folder,str(sim))
        #dframe.to_csv('main/final_tables/'+set_+str(sim)+'.csv')
        #end_set = time.time()
        #print("time to conclude set: {}".format(end_set-start_set))
        #plt.close('all')
