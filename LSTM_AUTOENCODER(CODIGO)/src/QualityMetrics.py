# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 13:51:37 2021

@author: debora
"""

import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score

def IdxShift(labcat):
    
    NWord=np.empty((0))
    NWord=np.append(NWord,0)
    for k in np.arange(len(labcat)):
        NWord=np.append(NWord,len(np.unique(labcat[k])))
        
    NWord=np.cumsum(NWord)    
    
    return NWord

def MetricSoA(y_true_idx, y_pred_idx,labcat):
    
    ThPaper=[5.5,1, 3.5,3 ,3, 3.1, 3,3.1]
    
    rec=[]
    prec=[]
    c_m=[]
    acc=[]
    nSamp=[]
    ncat=y_true_idx.shape[1]
    
    NWord=IdxShift(labcat)
    
    for k in np.arange(ncat):
        y_true=((y_true_idx[:,k]-NWord[k]+1)>ThPaper[k]).astype(int)
    #    print(np.unique(y_true_idx[:,k]-k*6+1))
        y_pred=((y_pred_idx[:,k]-NWord[k]+1)>ThPaper[k]).astype(int)
        
        nSamp.append(np.array([sum(1-y_true),sum(y_true)]))
       
        prec0,rec0,c_m0=ClassificationMetrics(
                y_true, y_pred)
        print(rec0)
        acc.append(accuracy_score(y_true, y_pred))
        prec.append(prec0)
        rec.append(rec0)
        c_m.append(c_m0)
        
    return prec,rec,c_m,acc,nSamp

def MetricSoAV0(y_true_idx, y_pred_idx,labcat):
    ThPaper=[5.5,1,3,3.1,3.5,3,3.1,3]
    rec=[]
    prec=[]
    c_m=[]
    acc=[]
    nSamp=[]
    ncat=y_true_idx.shape[1]
   
    
    for k in np.arange(ncat):
        y_true=((y_true_idx[:,k]-k*6+1)>ThPaper[k]).astype(int)
    #    print(np.unique(y_true_idx[:,k]-k*6+1))
        y_pred=((y_pred_idx[:,k]-k*6+1)>ThPaper[k]).astype(int)
        
        nSamp.append(np.array([sum(1-y_true),sum(y_true)]))
       
        prec0,rec0,c_m0=ClassificationMetrics(
                y_true, y_pred)
        print(rec0)
        acc.append(accuracy_score(y_true, y_pred))
        prec.append(prec0)
        rec.append(rec0)
        c_m.append(c_m0)
        
    return prec,rec,c_m,acc,nSamp

def HybridMetric(y_true_idx, y_pred_idx,labcat):
    
    # Categorical Attributes (Classification Metrics)
    nSamp=[]
    prec=[]
    rec=[]
    c_m=[]
    ncat=y_true_idx.shape[1]
    for k in np.arange(ncat):
        
        y_true=(np.round(y_true_idx[:,k]).flatten()).astype(int)
        y_pred=(np.round(y_pred_idx[:,k]).flatten()).astype(int)
    
  #  n_classes=max(np.max(y_true_idx[:,0:2]),np.max(y_pred_idx[:,0:2]))+1
        _,unique_counts=np.unique(np.array(y_true),return_counts=True)
        prec0,rec0,c_m0=ClassificationMetrics(
                y_true, y_pred, tags_categ=labcat[k])
        nSamp.append(unique_counts)
        prec.append(prec0)
        rec.append(rec0)
        c_m.append(c_m0)
    
    # Numeric Attributes (L2 Metrics)

    MSE= np.mean((y_true_idx-y_pred_idx)**2,axis=1)
    
    return prec,rec,c_m,nSamp,MSE

def HybridMetric_old(y_true_idx, y_pred_idx,CatVaridx,labcat):
    
    # Categorical Attributes (Classification Metrics)
    nSamp=[]
    prec=[]
    rec=[]
    c_m=[]
    
    for k in np.arange(len(labcat)):
        y_true=(np.round(y_true_idx[:,CatVaridx[k]]).flatten()).astype(int)
        y_pred=(np.round(y_pred_idx[:,CatVaridx[k]]).flatten()).astype(int)
    
  #  n_classes=max(np.max(y_true_idx[:,0:2]),np.max(y_pred_idx[:,0:2]))+1
        _,unique_counts=np.unique(np.array(y_true),return_counts=True)
        prec0,rec0,c_m0=ClassificationMetrics(
                y_true, y_pred, tags_categ=labcat[k])
        nSamp.append(unique_counts)
        prec.append(prec0)
        rec.append(rec0)
        c_m.append(c_m0)
    
    # Numeric Attributes (L2 Metrics)
    NoCatVaridx=np.setdiff1d(np.arange(y_true_idx.shape[1]),CatVaridx)
    y_true=np.array(y_true_idx[:,NoCatVaridx]).astype(float)
    y_pred=np.array(y_pred_idx[:,NoCatVaridx]).astype(float)
    MSE= np.mean((y_true-y_pred)**2,axis=1)
    
    return prec,rec,c_m,nSamp,MSE
    
def confusion_matrix_calculate(y_true, y_pred, tags_categ=None):

    if tags_categ is None:
        tags_categ = list(set(y_true))

  #  tags_ord = np.arange(len(tags_categ))
    cm = metrics.confusion_matrix(y_true, y_pred, labels=tags_categ, normalize='true')
    return cm

def ClassificationMetrics(y_true, y_pred, tags_categ=None):
    
    y_true=y_true.astype('int64')
    y_pred=y_pred.astype('int64')
    c_m = confusion_matrix_calculate(y_true, y_pred, tags_categ)
    prec,rec,_,_ = metrics.precision_recall_fscore_support(y_true, y_pred,labels=np.unique(y_true),
                                       zero_division=0)
    
    return prec,rec,c_m

def AUCMetrics(y_true, y_prob):
    
     n_classes=y_prob.shape[1]
     NSamp=y_prob.shape[0]
     y_gt = label_binarize(np.array(y_true).astype(int), classes=np.arange(n_classes))
     y_true=np.array(y_true).astype(int)

     if n_classes==2:
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob[:, 1])
        roc_auc = metrics.auc(fpr, tpr)
     else:
         fpr=[]
         tpr=[]
         thresholds=[]
         roc_auc=np.empty_like(y_gt)
         for i in range(y_gt.shape[1]):
             fpraux, tpraux, thresholdsaux= metrics.roc_curve(y_gt[:, i], y_prob[:, i])
             fpr.append(fpraux)
             tpr.append(tpraux)
             thresholds.append(thresholdsaux)
             roc_auc = metrics.auc(fpraux, tpraux)
    
     return roc_auc,fpr,tpr,thresholds