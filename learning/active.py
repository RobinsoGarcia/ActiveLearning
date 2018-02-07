import learning.datasets as dataset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity
from sklearn.datasets import fetch_20newsgroups, make_blobs,make_classification
import time
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
#%matplotlib inline
#plt.style.use('ggplot')

def delete_from_csr(mat, indices):

    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]

def random(sizeU,sample_size):
    idx=np.arange(sizeU)
    return np.random.choice(idx,sample_size,replace=False)

def least_confident(clf,U,sample_size,density_weights):
    prob=clf.predict_proba(U)
    lc = 1-np.sort(prob,axis=1)[:,-1]
    lc = np.multiply(lc,density_weights)
    return np.argsort(lc)[-sample_size:]

def margin(clf,U,sample_size,density_weights):
    prob=clf.predict_proba(U)
    margin = np.sort(prob,axis=1)[:,-2]-np.sort(prob,axis=1)[:,-1]
    margin = np.multiply(margin,density_weights)

    return np.argsort(margin)[-sample_size:]

def entropy(clf,U,sample_size,density_weights):
    prob = 0.99*clf.predict_proba(U)+0.01
    entropy = np.multiply(np.log(prob),prob)
    entropy = -np.sum(entropy,axis=1)
    entropy= np.multiply(entropy,density_weights)

    return np.argsort(entropy)[-sample_size:]

def qbc(X,y,X_,n,sample_size,clf,method,density_weights,qbc_sampling,labels):
    start = time.time()

    parameters={'n_estimators':[500,200,100,50,25],'learning_rate':[1,0.1]}
    clf1 = GridSearchCV(clf, parameters,scoring='accuracy',cv=3)
    #clf1=clf
    idxs=np.arange(X.shape[0])
    u=np.unique(y)

    predict = np.zeros((X_.shape[0],u.shape[0]))
    predictTeta=[]
    count=1
    for i in np.arange(n):
        min_len=0
        predictTeta.append([])

        while min_len<u.shape[0]:
            sample=np.random.choice(idxs,round(qbc_sampling*X.shape[0]))
            #min_len = len(np.unique(y[sample]))

        #sample=np.random.choice(idxs,round(qbc_sampling*X.shape[0]))
        clf1.fit(X[sample],y[sample])
        sampleU = clustering(X,y,X_,clf1,round(qbc_sampling*X.shape[0]),density_weights,labels)
        proba = clf1.predict_proba(X_[sampleU])
        predict = predict+proba
        predictTeta[i] =  proba
    predictC = predict/n

    if method=='entropy':
        vote = -np.sum(np.multiply(predictC,np.log(predictC)),axis=1)
    if method=='least_confident':
        vote =  1-np.sort(predictC,axis=1)[:,-1]
    if method=='margin':
        vote =  np.sort(predictC,axis=1)[:,-2]-np.sort(predictC,axis=1)[:,-1]
    elif method=='kl':
        kl_sum = np.empty((X_.shape[0],u.shape[0]))
        for i in np.arange(n):
            predictTeta_=predictTeta[i]
            kl_sum = kl_sum+np.multiply(predictTeta_,np.log(np.divide(predictC,predictTeta_)))
        vote=np.sum(kl_sum,axis=1)/n
    vote = np.multiply(vote,density_weights)
    end = time.time()
    print('time per qbc {}'.format(end-start))
    return np.argsort(vote)[-sample_size:]


def clustering(X,y,X_,clf,sample_size,density_weights,labels):

    #clf.fit(X,y)
    prob=clf.predict_proba(X_)
    margin = np.sort(prob,axis=1)[:,-2]-np.sort(prob,axis=1)[:,-1]
    margin = np.multiply(margin,density_weights)

    data = {'labels':labels,'margins':margin}

    data = pd.DataFrame.from_dict(data)

    score = data.groupby(by='labels').sum()
    cluster_label = int(score.idxmax())
    uncertain_cluster = data[data['labels']==cluster_label]
    vote = uncertain_cluster.sort_values(by='margins')[-sample_size:]
    return vote.index.tolist()





def bagging(X,y,X_,n,sample_size,clf,method,density_weights,qbc_sampling):
    start=time.time()

    clf1=clf
    clf1.fit(X,y)
 

    predictC = 0.99*clf1.predict_proba(X_)+0.01

    if method=='entropy':
        vote = -np.sum(np.multiply(predictC,np.log(predictC)),axis=1)
    if method=='least_confident':
        vote =  1-np.sort(predictC,axis=1)[:,-1]
    if method=='margin':
        vote =  np.sort(predictC,axis=1)[:,-2]-np.sort(predictC,axis=1)[:,-1]

    vote = np.multiply(vote,density_weights)
    end=time.time()
    print('bagging time: {} sec'.format(end-start))
    return np.argsort(vote)[-sample_size:]

def boosting(X,y,X_,n,sample_size,clf,method,density_weights,qbc_sampling):
    start=time.time()
 
    clf1=clf
    clf1.fit(X,y)
    #print(clf1.best_estimator_)
    predictC = 0.99*clf1.predict_proba(X_)+0.01

    if method=='entropy':
        vote = -np.sum(np.multiply(predictC,np.log(predictC)),axis=1)
    if method=='least_confident':
        vote =  1-np.sort(predictC,axis=1)[:,-1]
    if method=='margin':
        vote =  np.sort(predictC,axis=1)[:,-2]-np.sort(predictC,axis=1)[:,-1]

    vote = np.multiply(vote,density_weights)
    end=time.time()
    print('boosting time: {} sec'.format(end-start))
    return np.argsort(vote)[-sample_size:]

class active():
    def __init__(self,X=[],y=[],X_=[],y_=[],method='random',density_method='none',sparse=False):
        self.X=X
        self.y=y
        self.X_=X_
        self.y_=y_
        self.L=[]
        self.L_=[]
        self.U=[]
        self.U_=[]
        self.sparse=sparse
        self.clf = LogisticRegression(penalty='l2',C=1)
        self.sample_size=10
        self.method=method
        self.n_splits=3
        self.min_pool=20
        self.set_pool()
        self.iter=1
        self.sampled_Ubin = np.empty((0,self.U.shape[1]))
        self.sampled_U_bin =np.empty((0))
        self.accuracy_bin = np.array([])
        self.accuracy_train= np.array([])
        self.size_bin =np.array([])
        self.qbc_method='entropy'
        self.density_method=density_method
        self.set_density()
        self.qbc_sampling = 0.7
        self.clustering=0

    def set_pool(self):

        if self.sparse==True:
            min = self.min_pool
            num_samples=self.X.shape[0]
            idx = np.arange(num_samples)
            #idx = np.random.permutation(idx)
            self.L  = self.X[idx[:min]].toarray()
            self.L_ = self.y[idx[:min]]
            self.U  = self.X[idx[min:]]
            self.U_ = self.y[idx[min:]]
            self.sizeL=self.L.shape[0]
            self.sizeU=self.U.shape[0]
        else:
            min=self.min_pool
            num_samples=self.X.shape[0]
            idx = np.arange(num_samples)
            #idx = np.random.permutation(idx)
            self.L  = self.X[idx[:min]]
            self.L_ = self.y[idx[:min]]
            self.U  = self.X[idx[min:]]
            self.U_ = self.y[idx[min:]]
            self.sizeL=self.L.shape[0]
            self.sizeU=self.U.shape[0]

    def set_clusters(self):
        km = KMeans(n_clusters = 5)
        bundle = np.vstack((self.L,self.U))
        km.fit(bundle)
        self.cluster_labels = km.predict(self.U)

        '''
        min=round(self.min_pool/len(np.unique(self.y_)))
        num_samples=self.X.shape[0]
        idx = np.arange(num_samples)
        u,indices=np.unique(self.y,return_index=True)
        if self.sparse==True:
            self.L=self.X[indices].toarray()
            self.U=delete_from_csr(self.X,indices)
        else:
            self.L=self.X[indices]
            self.U=np.delete(self.X,indices,axis=0)

        self.L_=self.y[indices]
        self.U_=np.delete(self.y,indices,axis=0)

        for i in np.arange(min-1):

            u,indices=np.unique(self.U_,return_index=True)
            if self.sparse==True:
                self.L=np.append(self.L,self.U[indices].toarray(),axis=0)
            else:
                self.L=np.append(self.L,self.U[indices],axis=0)

            self.L_=np.append(self.L_,self.U_[indices])
            if self.sparse==True:
                self.U= delete_from_csr(self.U,indices)
            else:
                self.U=np.delete(self.U,indices,axis=0)
            self.U_=np.delete(self.U_,indices,axis=0)
        '''

        return self.L,self.L_,self.U,self.U_

    def fit(self):
        iter=self.iter

        for i in np.arange(self.iter):
            self.clf.fit(self.L,self.L_)
            self.query()
            self.update()
            self.accuracy_bin = np.append(self.accuracy_bin,self.accuracy())
            self.accuracy_train = np.append(self.accuracy_train,self.train_accuracy())

            self.size_bin = np.append(self.size_bin,self.sizeL)
        #plt.plot(self.size_bin,1-self.accuracy_bin,label='test_error')
        #plt.plot(self.size_bin,1-self.accuracy_test,label='train_error')
        #plt.legend()

    def update(self):
        idx=self.idx

        if self.sparse==True:
            self.L = np.vstack((self.L,self.U[idx].toarray()))
        else:
            self.L = np.vstack((self.L,self.U[idx]))
        self.L_ = np.append(self.L_,self.U_[idx])

        if self.sparse==True:
            self.sampled_Ubin = np.append(self.sampled_Ubin,self.U[idx].toarray(),axis=0)
        else:
            self.sampled_Ubin = np.append(self.sampled_Ubin,self.U[idx],axis=0)
        self.sampled_U_bin = np.append(self.sampled_U_bin ,self.U_[idx],axis=0)

        if self.sparse==True:
            self.U = delete_from_csr(self.U,idx)
        else:
            self.U = np.delete(self.U,idx,axis=0)

        self.U_ = np.delete(self.U_,idx,axis=0)
        self.sizeU = self.U.shape[0]
        self.sizeL = self.L.shape[0]
        self.density_weights = np.delete(self.density_weights,idx,axis=0)
        if self.clustering==1:
            self.set_clusters()
        pass

    def query(self):

        if self.method=='random':
            self.idx = random(self.U.shape[0],self.sample_size)

        elif self.method=='least_confident':
            self.idx = least_confident(self.clf,self.U,self.sample_size,self.density_weights)

        elif self.method == 'margin':
            self.idx = margin(self.clf,self.U,self.sample_size,self.density_weights)

        elif self.method == 'qbc':
            self.set_clusters()
            self.idx = qbc(self.L,self.L_,self.U,self.n_splits,self.sample_size,AdaBoostClassifier(base_estimator=self.clf),self.qbc_method,self.density_weights,self.qbc_sampling,self.cluster_labels)

        elif self.method == 'entropy':
            self.idx = entropy(self.clf,self.U,self.sample_size,self.density_weights)

        elif self.method == 'bagging':
            self.idx = bagging(self.L,self.L_,self.U,self.n_splits,self.sample_size,RandomForestClassifier(n_estimators=500,max_depth=3),self.qbc_method,self.density_weights,self.qbc_sampling)

        elif self.method == 'boosting':
            est=init(LogisticRegression(penalty='l2',C=1))
            self.idx = boosting(self.L,self.L_,self.U,self.n_splits,self.sample_size,GradientBoostingClassifier(max_depth=1,n_estimators=1000,subsample=0.3,warm_start=True,learning_rate=0.1),self.qbc_method,self.density_weights,self.qbc_sampling)
        elif self.method ==  'clustering':
            self.set_clusters()
            self.idx = clustering(self.L,self.L_,self.U,self.clf,self.sample_size,self.density_weights,self.cluster_labels)



    def accuracy(self):
        predict = self.clf.predict(self.X_)
        accuracy = (predict==self.y_).mean()
        #print(accuracy)
        return accuracy

    def train_accuracy(self):
        predict = self.clf.predict(self.X)
        accuracy = (predict==self.y).mean()
        #print(accuracy)
        return accuracy

    def set_density(self):
        if self.density_method=='euclidean':
            self.density_weights=np.sum(euclidean_distances(self.U,self.U),axis=1)/self.U.shape[0]
        elif self.density_method=='cosine':
            self.density_weights=np.sum(cosine_similarity(self.U,self.U),axis=1)/self.U.shape[0]
        else:
            self.density_weights=np.ones(self.U.shape[0])

def get_index(a,b):
    count=1
    bin_=[]
    for i in np.arange(len(a)):
        count=count*a[i]/b[i]
        bin_.append(count)
    return bin_

def get_density(X,y):
    kernel = KernelDensity(kernel='gaussian',bandwidth=0.2)
    kernel.fit(X)

    xmin=X[:,0].min()
    xmax=X[:,0].max()
    ymin=X[:,1].min()
    ymax=X[:,1].max()

    plt.figure()
    a,b = np.mgrid[xmin:xmax:0.1,ymin:ymax:0.1]
    positions = np.vstack([a.ravel(),b.ravel()])
    z=np.reshape(np.exp(kernel.score_samples(positions.T)),a.shape)
    cnt = plt.contour(a,b,z,50,cmap=plt.cm.gray)

    return [xmin,xmax,ymin,ymax]

def get_area(x):
    area=0
    area200=0
    for i in np.arange(x.shape[0]-1):
        area = area+(x[i+1]+x[i])/2

    for i in np.arange(200-1):
        area200 = area200+(x[i+1]+x[i])/2

    return area,area200

class init:
    def __init__(self, est):
        self.est = est
    def predict(self, X):
        return self.est.predict_proba(X)#[:,1][:,np.newaxis]
    def fit(self, X, y,w):
        self.est.fit(X, y,w)
