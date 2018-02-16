from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups, make_blobs,make_classification
from sklearn.datasets import load_digits,load_iris,load_breast_cancer,load_wine
import numpy as np
from numpy.random import randint,rand
import os
import pandas as pd
from pandas.tools.plotting import andrews_curves
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import learning
#%matplotlib inline
#plt.style.use('ggplot')

hmdr = os.path.join(learning.__path__[0],'UCI_datasets/')

def get_blob(r,std):
    A,b= make_blobs(n_samples=2000,centers=5,cluster_std=std,n_features=2,random_state=r)
    split=0.7
    size=A.shape[0]
    idx = np.arange(size)
    #idx = np.random.permutation(idx)
    X = A[idx[:round(split*size)]]
    X = np.insert(X,0,1.0,axis=1)
    y = b[idx[:round(split*size)]]
    X_= A[idx[round(split*size)]:]
    X_ = np.insert(X_,0,1.0,axis=1)
    y_= b[idx[round(split*size)]:]
    return X,y,X_,y_

def get_classi(r,n_informative):
    A,b= make_classification(n_samples=2000,n_features=20,n_informative=n_informative,random_state=r)

    split=0.7
    size=A.shape[0]
    idx = np.arange(size)
    #idx = np.random.permutation(idx)
    X = A[idx[:round(split*size)]]
    X = np.insert(X,0,1.0,axis=1)
    y = b[idx[:round(split*size)]]
    X_= A[idx[round(split*size)]:]
    X_ = np.insert(X_,0,1.0,axis=1)
    y_= b[idx[round(split*size)]:]
    return X,y,X_,y_


def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))
    return newsgroups_train, newsgroups_test

def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer(max_df=0.8
                            ,strip_accents='unicode'
                            ,lowercase=True
                            ,ngram_range=(1,1)
                            ,norm='l2'
                            ,stop_words='english'
                            )
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    vocab = tf_idf_vectorize.decode(feature_names)
    shape = tf_idf_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[tf_idf_train.sum(axis=0).argmax()]))
    return tf_idf_train, tf_idf_test, feature_names

def transform(X, y=None):
    X=[re.sub("[^a-zA-Z]", " ", x) for x in X]
    X=[re.sub(r'(.)\1+', r'\1', x) for x in X]
    X=[re.sub(r'\W*\b\w{1,3}\b'," ", x) for x in X]
    return X


def get_andrews(X,y):
    data = pd.DataFrame(X)
    data['labels']=y
    andrews_curves(data, 'labels')

def load(dataset):
    if dataset=="20news":
        from scipy.sparse import coo_matrix,csr_matrix, hstack
        train_data, test_data = load_data()
        X,X_,feature_names=tf_idf_features(train_data, test_data)
        y=train_data.target
        y_=test_data.target

        #get_andrews(X.toarray(),y)
        #plt.title('Andrews_curves: 20news')
        #plt.savefig('results/andrews/20news')

        ones = coo_matrix(np.ones(X.shape[0]))
        X = csr_matrix(hstack((X,ones.T)))
        ones_ = coo_matrix(np.ones(X_.shape[0]))
        X_=csr_matrix(hstack((X_,ones_.T)))


        return X,y,X_,y_
    elif dataset=='digits':
        data=load_digits()
        size=data.data.shape[0]
        count=0
        values,c= np.unique(data.target,return_counts=True)

        while count!=1:
            idx=np.arange(size)
            idx=np.random.permutation(idx)

            #idx = np.load('seeds/wine.npy')
            split=0.7

            X = data.data[idx[:round(split*size)]]
            y = data.target[idx[:round(split*size)]]
            X_= data.data[idx[round(split*size)]:]
            y_=data.target[idx[round(split*size)]:]
            yvalues,ycounts=np.unique(y,return_counts=True)
            y_values,y_counts=np.unique(y_,return_counts=True)
            if len(yvalues)==len(values):
                if len(y_values)==len(values):
                    count = 1
        X = np.insert(X,0,1.0,axis=1)
        X_ = np.insert(X_,0,1.0,axis=1)

        return X,y,X_,y_
    elif dataset=='breast_cancer':
        data=load_breast_cancer()
        size=data.data.shape[0]
        count=0
        values,c= np.unique(data.target,return_counts=True)

        while count!=1:
            idx=np.arange(size)
            idx=np.random.permutation(idx)

            #idx = np.load('seeds/wine.npy')
            split=0.7

            X = data.data[idx[:round(split*size)]]
            y = data.target[idx[:round(split*size)]]
            X_= data.data[idx[round(split*size)]:]
            y_=data.target[idx[round(split*size)]:]
            yvalues,ycounts=np.unique(y,return_counts=True)
            y_values,y_counts=np.unique(y_,return_counts=True)
            if len(yvalues)==len(values):
                if len(y_values)==len(values):
                    count = 1

        X = np.insert(X,0,1.0,axis=1)
        X_ = np.insert(X_,0,1.0,axis=1)

        return X,y,X_,y_

    elif dataset=='wine':

        data=load_wine()
        size=data.data.shape[0]
        count=0
        values,c= np.unique(data.target,return_counts=True)

        while count!=1:
            idx=np.arange(size)
            idx=np.random.permutation(idx)

            #idx = np.load('seeds/wine.npy')
            split=0.7

            X = data.data[idx[:round(split*size)]]
            y = data.target[idx[:round(split*size)]]
            X_= data.data[idx[round(split*size)]:]
            y_=data.target[idx[round(split*size)]:]
            yvalues,ycounts=np.unique(y,return_counts=True)
            y_values,y_counts=np.unique(y_,return_counts=True)
            if len(yvalues)==len(values):
                if len(y_values)==len(values):
                    count = 1


        #get_andrews(X,y)
        #plt.title('Andrews_curves: wine')
        #plt.savefig('results/andrews/wine')

        X = np.insert(X,0,1.0,axis=1)
        X_ = np.insert(X_,0,1.0,axis=1)


        return X,y,X_,y_
    elif dataset=='iris':
        data=load_iris()
        size=data.data.shape[0]
        count=0
        values,c= np.unique(data.target,return_counts=True)

        while count!=1:
            idx=np.arange(size)
            idx=np.random.permutation(idx)

            #idx = np.load('seeds/wine.npy')
            split=0.7

            X = data.data[idx[:round(split*size)]]
            y = data.target[idx[:round(split*size)]]
            X_= data.data[idx[round(split*size)]:]
            y_=data.target[idx[round(split*size)]:]
            yvalues,ycounts=np.unique(y,return_counts=True)
            y_values,y_counts=np.unique(y_,return_counts=True)
            if len(yvalues)==len(values):
                if len(y_values)==len(values):
                    count = 1
        X = np.insert(X,0,1.0,axis=1)
        X_ = np.insert(X_,0,1.0,axis=1)

        return X,y,X_,y_

    elif dataset=='pama-indians':
        load=np.loadtxt(hmdr+'/pama_indians.csv',delimiter=",")
        target = load[:,load.shape[1]-1]
        data = load[:,:load.shape[1]-1]
        count=0
        size=data.data.shape[0]
        values,c= np.unique(target,return_counts=True)

        while count!=1:
            idx=np.arange(size)
            idx=np.random.permutation(idx)

            #idx = np.load('seeds/wine.npy')
            split=0.7

            X = data[idx[:round(split*size)]]
            y = target[idx[:round(split*size)]]
            X_= data[idx[round(split*size)]:]
            y_= target[idx[round(split*size)]:]
            yvalues,ycounts=np.unique(y,return_counts=True)
            y_values,y_counts=np.unique(y_,return_counts=True)
            if len(yvalues)==len(values):
                if len(y_values)==len(values):
                    count = 1
        X = np.insert(X,0,1.0,axis=1)
        X_ = np.insert(X_,0,1.0,axis=1)

        return X,y,X_,y_

    elif dataset=='liver':
        load=np.loadtxt(hmdr+'liver.csv',delimiter=",")
        target = load[:,load.shape[1]-1]
        data = load[:,:load.shape[1]-1]

        size=data.data.shape[0]
        count=0
        values,c= np.unique(target,return_counts=True)

        while count!=1:
            idx=np.arange(size)
            idx=np.random.permutation(idx)

            #idx = np.load('seeds/wine.npy')
            split=0.7

            X = data[idx[:round(split*size)]]
            y = target[idx[:round(split*size)]]
            X_= data[idx[round(split*size)]:]
            y_=target[idx[round(split*size)]:]
            yvalues,ycounts=np.unique(y,return_counts=True)
            y_values,y_counts=np.unique(y_,return_counts=True)
            if len(yvalues)==len(values):
                if len(y_values)==len(values):
                    count = 1

        X = np.insert(X,0,1.0,axis=1)
        X_ = np.insert(X_,0,1.0,axis=1)

        return X,y,X_,y_


    elif dataset=='crx':
        load=pd.read_csv(hmdr+'/crx.csv')
        load.dtypes
        load = pd.get_dummies(load)
        target = load['p'].as_matrix()
        data = load.drop(['p'],axis=1).as_matrix()
        count=0
        size=data.shape[0]
        values,c= np.unique(target,return_counts=True)


        while count!=1:
            idx=np.arange(size)
            idx=np.random.permutation(idx)
            type(idx[1])
            #idx = np.load('seeds/wine.npy')
            split=0.7

            X = data[idx[:round(split*size)]]
            y = target[idx[:round(split*size)]]
            X_= data[idx[round(split*size)]:]
            y_=target[idx[round(split*size)]:]
            yvalues,ycounts=np.unique(y,return_counts=True)
            y_values,y_counts=np.unique(y_,return_counts=True)
            if len(yvalues)==len(values):
                if len(y_values)==len(values):
                    count = 1

        X = np.insert(X,0,1.0,axis=1)
        X_ = np.insert(X_,0,1.0,axis=1)



        return X,y,X_,y_


    elif dataset=='tictac':
        load=pd.read_csv(hmdr+'/tictac.csv')
        load.dtypes
        load = pd.get_dummies(load)
        target = load['p'].as_matrix()
        data = load.drop(['p'],axis=1).as_matrix()
        values,c= np.unique(target,return_counts=True)

        size=data.shape[0]
        count=0
        while count!=1:
            idx=np.arange(size)
            idx=np.random.permutation(idx)

            #idx = np.load('seeds/wine.npy')
            split=0.7

            X = data[idx[:round(split*size)]]
            y = target[idx[:round(split*size)]]
            X_= data[idx[round(split*size)]:]
            y_= target[idx[round(split*size)]:]
            yvalues,ycounts=np.unique(y,return_counts=True)
            y_values,y_counts=np.unique(y_,return_counts=True)
            if len(yvalues)==len(values):
                if len(y_values)==len(values):
                    count = 1

        X = np.insert(X,0,1.0,axis=1)
        X_ = np.insert(X_,0,1.0,axis=1)

        return X,y,X_,y_
