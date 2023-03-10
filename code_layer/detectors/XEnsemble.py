import numpy as np
import pickle
import os
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import sklearn
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import scipy

def auc_curve(y,prob):
    fpr,tpr,thresholds = roc_curve(y,prob) ###计算真正率和假正率
    roc_auc = sklearn.metrics.auc(fpr,tpr) ###计算auc的值
    return roc_auc,thresholds

def distribution_area(X,l_lim,r_lim):
    if isinstance(X,np.ndarray):
        X.sort()
    elif isinstance(X,list):
        X.sort()
        X = np.array(X)
    else:
        raise Exception("distribution_area 的 X 类型不对 {}".format(type(X)))
    if X.min()<l_lim or X.max()>r_lim:
        raise Exception("distribution_area 的 X 范围不对 {}~{} not in {}~{}".format(X.min(),X.max(),l_lim,r_lim))
    sz = X.shape[0]
    X = np.append(X,r_lim)
    total_area = 0
    for i in range(sz):
        h = (i+1)/sz
        v = X[i+1]-X[i]
        total_area+=h*v 
    return total_area

def distribution_distance(X0,X1):
    # l_lim = 1e100
    # r_lim = -1e100
    # for item in X0:
    #     l_lim = min(l_lim,item)
    #     r_lim = max(r_lim,item)
    # for item in X1:
    #     l_lim = min(l_lim,item)
    #     r_lim = max(r_lim,item)
    # return distribution_area(X0,l_lim,r_lim) - distribution_area(X1,l_lim,r_lim)
    hist, bin_edges = np.histogram(X0,bins=1000,density=True) 
    hist2, bin_edges2 = np.histogram(X1,bins=1000,density=True) 
    dis = scipy.stats.wasserstein_distance(bin_edges[:-1],bin_edges2[:-1],hist,hist2)
    return dis


def make_dir(path):
    try:
        os.makedirs(path)
    except Exception as e:
        print(f'{path}  already exist')

def pk_dump(data,filename):
    with open(filename,'wb') as f:
        pickle.dump(data, f)

def pk_load(filename):
    with open(filename,'rb') as f:
        data = pickle.load(f)
    return data

class FFT_Reducer_C32:
    def __init__(self):
        # self.pool = torch.nn.AvgPool2d(2) # kernel shape = (7,7)  stride = (7,7)
        pass
    
    def train(self,X:torch.Tensor):
        return self.reduce(X)

    def reduce(self,X:torch.Tensor):
        shape = X.shape
        print(X.shape)
        output = torch.fft.fft2(X,dim=(-2,-1))
        real = output.real[:,:,95:128,95:128]
        imag = output.imag[:,:,95:128,95:128]
        output = torch.stack((real,imag),-1)
        output = output.view(shape[0],-1)
        print(output.shape)
        return output
    
    def dump(self,path):
        pass

    def load(self,path):
        pass

class PCA_Reducer:
    def __init__(self,use_sklearn=False,n_components=1000):
        self.eigen_vecs = None
        self.mean = None
        self.use_sklearn = use_sklearn
        self.model=None
        self.n_components = n_components
        print(f"use_sklearn={use_sklearn}")
    
    def train(self,X:torch.Tensor):
        size = X.shape[0]
        X0 = X
        X = X.view(size,-1).numpy()
        if self.use_sklearn:
            self.model = sklearn.decomposition.PCA(n_components=self.n_components,svd_solver ='full')
            self.model.fit(X)
            return self.model.transform(X)
        try:
            # Data matrix X, assumes 0-centered
            n, m = X.shape
            self.mean = X.mean(axis=0)
            X = X - X.mean(axis=0)
            # Compute covariance matrix
            C = np.dot(X.T, X) / (n-1)
            # C = np.cov(X.T,ddof=0)
            # Eigen decomposition
            eigen_vals, eigen_vecs = np.linalg.eig(C)
            # Project X onto PC space
            self.eigen_vecs = eigen_vecs
            X_pca = np.dot(X, eigen_vecs)
            return X_pca
        except:
            self.use_sklearn=True
            print("use svd in sklearn")
            return self.train(X0)

    def reduce(self,X):
        size = X.shape[0]
        X = X.view(size,-1).numpy()
        if self.use_sklearn:
            return self.model.transform(X)
        return np.dot(X-self.mean,self.eigen_vecs)
    
    def dump(self,path):
        pk_dump([self.eigen_vecs,self.mean,self.use_sklearn,self.model],path)

    def load(self,path):
        try:
            self.eigen_vecs,self.mean,self.use_sklearn,self.model = pk_load(path)
        except:
            self.eigen_vecs,self.mean = pk_load(path)
            self.use_sklearn=False


class PCA_Reducer_T500(PCA_Reducer):
    def __init__(self):
        # self.pool = torch.nn.AvgPool2d(2) # kernel shape = (7,7)  stride = (7,7)
        super().__init__(use_sklearn=True)
        pass
    
    def train(self,X:torch.Tensor):
        return super().train(X)[:,500:]

    def reduce(self,X:torch.Tensor):
        return super().reduce(X)[:,500:]


class FFT_PCA_Reducer(PCA_Reducer):
    def __init__(self,n_components=1000):
        # self.pool = torch.nn.AvgPool2d(2) # kernel shape = (7,7)  stride = (7,7)
        super().__init__(use_sklearn=True,n_components=n_components)
    
    def train(self,X):
        output = torch.fft.fft2(X,dim=(-2,-1))
        real = output.real
        imag = output.imag
        output = torch.stack((real,imag),-1)
        return super().train(output)

    def reduce(self,X):
        output = torch.fft.fft2(X,dim=(-2,-1))
        real = output.real
        imag = output.imag
        output = torch.stack((real,imag),-1)
        return super().reduce(output)


class Normalizer:
    def __init__(self):
        self.mean = None 
        self.max = None 
        self.min = None 
        self.std = None 
    
    def fit(self,x):
        self.max = np.max(x)
        self.min = np.min(x)
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
    
    def get(self,x,method = 'min-max',reverse=False):
        if reverse:
            x = 2*self.mean - x
        if method == 'min-max':
            # print('norm with min-max')
            return (x-self.min)/(self.max-self.min)
        else:
            # print('norm with mean-std')
            return (x-self.mean)/self.std
    def dump(self,path):
        make_dir(path)
        pk_dump(self.min,os.path.join(path,'min.pth'))
        pk_dump(self.max,os.path.join(path,'max.pth'))
        pk_dump(self.mean,os.path.join(path,'mean.pth'))
        pk_dump(self.std,os.path.join(path,'std.pth'))
    
    def load(self,path):
        self.max = pk_load(os.path.join(path,'max.pth'))
        self.min = pk_load(os.path.join(path,'min.pth'))
        self.mean = pk_load(os.path.join(path,'mean.pth'))
        self.std = pk_load(os.path.join(path,'std.pth'))

import numpy as np
import pandas as pd
class Ensembler:
    def __init__(self,interpret_methods,detect_method='iforest',use_normers=True,use_voting=False):
        self.interpret_methods = interpret_methods
        self.detect_method = detect_method
        self.use_voting = use_voting
        self.clf_list = []
        self.result = {}
        self.normers = {}
        self.use_normers=use_normers
        rng = np.random.RandomState(1587)
        for i_method in interpret_methods:
            self.normers[i_method] = Normalizer()
            if detect_method == 'LOF':
                clf = LocalOutlierFactor(n_neighbors=20,novelty=True)
            elif detect_method == 'SVDD':
                clf = OneClassSVM(kernel='rbf', gamma='auto', shrinking=True, cache_size=200, verbose=False,max_iter=-1)
            elif detect_method == 'iforest':
                clf = IsolationForest(n_estimators=500,max_samples=0.8, max_features=0.5, random_state=rng,warm_start=True)
            elif detect_method == 'Envelope':
                clf = EllipticEnvelope()
            else:
                raise Exception(f"{detect_method} not implemented")
            self.clf_list.append(clf)
        
        self.iso_forest_ensembler = None
    
    def save(self,dir_path):
        make_dir(dir_path)
        for i_method,clf in zip(self.interpret_methods,self.clf_list):
            pk_dump(clf,os.path.join(dir_path,f'{i_method}.pth'))
        if self.use_normers:
            for interpret_method in self.interpret_methods:
                normalizer = self.normers[interpret_method]
                normalizer.dump(os.path.join(dir_path,f'normer/{interpret_method}'))
        pk_dump(self.iso_forest_ensembler,os.path.join(dir_path,f'iso_forest_ensembler.pth'))
        
    def load(self,dir_path):
        for ind,i_method in enumerate(self.interpret_methods):
            clf = pk_load(os.path.join(dir_path,f'{i_method}.pth'))
            self.clf_list[ind] = clf
        if self.use_normers:
            normers = {}
            for interpret_method in self.interpret_methods:
                normalizer = Normalizer()
                normalizer.load(os.path.join(dir_path,f'normer/{interpret_method}'))
                normers[interpret_method] = normalizer
            self.normers = normers
        self.iso_forest_ensembler = pk_load(os.path.join(dir_path,f'iso_forest_ensembler.pth'))

    def fit(self,interpret_method,X_train):
        print('fit',interpret_method)
        for i_method,clf in zip(self.interpret_methods,self.clf_list):
            if i_method == interpret_method:
                clf.fit(X_train)
                if self.use_voting:
                    return
                if self.use_normers:
                    Z = clf.decision_function(X_train)
                    self.normers[i_method].fit(Z)
                break
    
    def calculate(self,interpret_method,X,norm_method='min-max',reverse=False):
        for i_method,clf in zip(self.interpret_methods,self.clf_list):
            if i_method == interpret_method:
                if self.use_voting:
                    Z = clf.predict(X)
                    self.result[i_method] = Z
                    break
                Z = clf.decision_function(X)
                if self.use_normers and self.normers is not None:
                    Z = self.normers[i_method].get(Z,method=norm_method,reverse=reverse)
                self.result[i_method] = Z
                break

    def sub_detector(self,interpret_method):
        for i_method,clf in zip(self.interpret_methods,self.clf_list):
            if i_method == interpret_method:
                return self.result[i_method]

    def find_clf(self,interpret_method):
        for i_method,clf in zip(self.interpret_methods,self.clf_list):
            if i_method == interpret_method:
                return clf

    def fit_forest_ensembler(self):
        Zv=[np.array(r) for k,r in self.result.items()]
        Z = np.vstack(Zv).T
        self.iso_forest_ensembler = IsolationForest(n_estimators=200,max_samples=0.8, max_features=0.8,warm_start=True)
        self.iso_forest_ensembler.fit(Z)

    def calculate_forest_ensembler(self):
        Zv=[np.array(r) for k,r in self.result.items()]
        Z = np.vstack(Zv).T
        return self.iso_forest_ensembler.decision_function(Z)

    def ensemble(self,method='min'):
        for k,v in self.result.items():
            length = len(v)
            break

        if "min" in method:
            Z = []
            for ind in range(length):
                _min = 1e10
                for k,r in self.result.items():
                    if r[ind]<_min:
                        _min = r[ind]
                Z.append(_min)
            Z = np.array(Z)
            return Z
        elif "max" in method:
            Z = []
            for ind in range(length):
                _max = -1e10
                for k,r in self.result.items():
                    if r[ind]>_max:
                        _max = r[ind]
                Z.append(_max)
            Z = np.array(Z)
            return Z
        elif "sum" in method:
            Z = []
            for ind in range(length):
                tmp = 0
                for k,r in self.result.items():
                    tmp += r[ind]
                Z.append(tmp)
            Z = np.array(Z)
            return Z
        elif "iforest" in method:
            if self.iso_forest_ensembler is None:
                self.fit_forest_ensembler()
            return self.calculate_forest_ensembler()

    def save_z(self,Y,path):
        size = Y.shape[0]
        df = {}
        for ind in range(size):
            item = {}
            for k,r in self.result.items():
                item[k] = r[ind]
            item['Y'] = Y[ind]
            df[ind]=item
        df = pd.DataFrame.from_dict(df)
        df.to_csv(path,float_format = '%.3f')

    def get_dist_distance(self,X_org,X_adv):
        cols = X_org.shape[1]
        cols = [i for i in range(cols)]
        dist = []
        for c in cols:
            dist.append(distribution_distance(X_org[:,c],X_adv[:,c]))
        return cols,dist

    def draw_dist_distance(self,X_org,X_adv,img_path):
        plt.figure()
        cols,dist = self.get_dist_distance(X_org,X_adv)
        plt.bar(cols,dist)
        plt.savefig(img_path)
        plt.close()
        return dist

    def draw_hist_col(self,X_org,X_adv,col,img_path):
        plt.figure()
        plt.hist(X_org[:,col],bins=1000,density=True)
        plt.hist(X_adv[:,col],bins=1000,density=True)
        plt.savefig(img_path)
        plt.close()
    
    def draw_point(self,X_org,X_adv,col0,col1,img_path):
        plt.figure()
        plt.plot(X_org[:,col0],X_org[:,col1],'o')
        plt.plot(X_adv[:,col0],X_adv[:,col1],'x')
        plt.savefig(img_path)
        plt.close()

    def draw_all_auc(self,X,Y,img_path):
        all_auc = []
        cols = [i for i in range(X.shape[1])]
        for col in cols:
            XX = X[:,col]
            roc_auc,thresholds = auc_curve(Y,XX)
            all_auc.append(np.abs(roc_auc-0.5))
        plt.bar(cols,all_auc)
        plt.savefig(img_path)
