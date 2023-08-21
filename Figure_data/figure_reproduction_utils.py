#!/usr/bin/env python
# coding: utf-8

import numpy as np
from numpy import trapz
import matplotlib.pyplot as plt
from io import StringIO
import re
import pymatgen as mg
import tensorflow as tf
import random
import os
from pymatgen.core.periodic_table import Element
import json
import linecache
from pymatgen.core import Composition
import scipy
from scipy.stats import *

def balanced_frac(materials, ox_dict):
    balanced = []
    for material in materials:
        try:
            ox_guesses = list(Composition(material).oxi_state_guesses(oxi_states_override=ox_dict))
            if (len(ox_guesses) > 0):
                for j in range(len(ox_guesses)):
                    success=0
                    for key in (ox_guesses[j]):
                        coeff=(ox_guesses[j][key])
                        if(coeff.is_integer()):
                            success=success+1
                        else:
                            continue
                    if(success==len(ox_guesses[j])):
                        #print(ox_guesses[j])
                        return(['True', ox_guesses[j]])
            if(len(Composition(material))==1):
                return(['True','0'])
            else:
                balanced.append(0)
            #print(str(material), len(material))
        except:
            balanced.append(0)
    return(['False', 'None'])

def balanced_frac2(materials, ox_dict):
    balanced=[]
    for material in materials:
        try:
            ox_guesses = list(Composition(material).oxi_state_guesses(oxi_states_override=common_ox_dict))
            if (len(ox_guesses) > 0):
                for j in range(len(ox_guesses)):
                    success=0
                    for key in (ox_guesses[j]):
                        coeff=(ox_guesses[j][key])
                        if(coeff.is_integer()):
                            success=success+1
                        else:
                            continue
                    if(success==len(ox_guesses[j])):
                        #print(ox_guesses[j])
                        balanced.append([material,'True', ox_guesses[j]])
                    else:
                        balanced.append([material,'False','0'])
            elif(len(Composition(material))==1):
                balanced.append([material,'True','0'])
            else:
                balanced.append([material,'False','0'])
        except:
            balanced.append([material,'False','0'])
    return(balanced)

def get_weighted_F1(TP,FP,TN,FN,num_positive,num_negative):
    num_positive_class=num_positive/(num_positive+num_negative)
    num_negative_class=num_negative/(num_positive+num_negative)
    pos_prec=TP/(TP+FP)
    pos_rec=TP/(TP+FN)
    neg_prec=TN/(TN+FN)
    neg_rec=TN/(TN+FP)
    F1_pos=2*(pos_prec*pos_rec)/(pos_prec+pos_rec)
    F1_neg=2*(neg_prec*neg_rec)/(neg_prec+neg_rec)
    F1_weighted=(num_positive_class*F1_pos) + (num_negative_class*F1_neg)
    return(F1_weighted)

def get_batch(batch_size, neg_positive_ratio, use_semi_weights, model_name, seed=False, seed_value=0):
    def random_lines(filename, file_size, num_samples):
        idxs = random.sample(range(1,file_size), num_samples)
        return ([linecache.getline(filename, i) for i in idxs], idxs)
    if(seed):
        random.seed(seed_value)
        np.random.seed(seed_value)
    else:
        random.seed()
        np.random.seed()
    num_positive_examples=int(np.floor(batch_size*(1/(1+neg_positive_ratio))))
    num_negative_examples=batch_size-num_positive_examples
    noTr_positives=48200 #number positive examples in train set
    noTr_negatives=noTr_positives*neg_positive_ratio #no. negatives examples in train set
    noTr = noTr_positives + (noTr_negatives) #total size of train set
    #only sample from first 90% of dataset; shuffle first
    data1=[]
    pulled_lines1,idxs1=random_lines('./Datasets/icsd_full_data_unique_no_frac_no_penta_2020.txt', noTr_positives,num_positive_examples)
    for line in pulled_lines1:
        data1.append(line.replace('\n',''))
    data0=[]
    pulled_lines0,idxs0=random_lines('./Datasets/full_unsynthesized_examples.txt', noTr_negatives, num_negative_examples)
    for line in pulled_lines0:
        data0.append(line.replace('\n',''))
    #do consistent shuffling once examples have been chosen
    random.seed(3)
    np.random.seed(3)
    #shuffle the positive and negative examples with themselves
    negative_indices=list(range(0,len(data0)))
    random.shuffle(negative_indices)
    positive_indices=list(range(0,len(data1)))
    random.shuffle(positive_indices)
    data0=np.array(data0)
    data1=np.array(data1)
    idxs0=np.array(idxs0)
    idxs1=np.array(idxs1)
    data0=data0[negative_indices]
    data1=data1[positive_indices]
    featmat0=get_features(data0)
    featmat1=get_features(data1)
    idxs0=idxs0[negative_indices]
    idxs1=idxs1[positive_indices]
    #get labels
    labs = np.zeros((len(data0) + len(data1),1))
    for ind,ent in enumerate(data1):
        labs[ind,0] = 1
    unique, counts = np.unique(labs, return_counts=True)
    ind0 = np.where(labs==0)[0] #indices of label=0
    ind1 = np.where(labs==1)[0] #indices of label=1
    #combine positives and negatives and shuffle
    featmat3 = np.concatenate((featmat0,featmat1)) #set legths of labels 0 and 1 to be the same in the new feature matrix featmat3
    datasorted = np.concatenate((data0,data1)) #data ordered the same as featmat3
    labs3 = np.concatenate((labs[ind0], labs[ind1]), axis=0) #labels ordered the same as featmat3
    idxs_full=np.concatenate((idxs0,idxs1))
    noS = len(featmat3)
    ind = list(range(0,noS)) #training set index
    random.shuffle(ind) #shuffle training set index
    #indB = list(range(0,noTr_subset)) #used later for batch
    labs3 = np.column_stack((labs3,np.abs(labs3-1)))
    xtr_batch = featmat3[ind[0:],:]
    ytr_batch = labs3[ind[0:],:]
    data_batch=datasorted[ind[0:]]
    idxs_full=idxs_full[ind[0:]]
    #all weights stuff here
    weights_full=[]
    if(use_semi_weights):
        weights1=[]
        file=open('semi_weights_testing_pos_20M' + model_name + '.txt','r')
        content=file.readlines()
        weights1=[]
        for i in idxs1:
            weights1.append(float(content[i-1].split()[1]))
        file.close()
        weights0=[]
        file=open('semi_weights_testing_neg_20M' + model_name + '.txt','r')
        content=file.readlines()
        for i in idxs0:
            weights0.append(float(content[i-1].split()[1]))
        file.close()
        weights0=np.array(weights0)
        weights1=np.array(weights1)
        weights0=weights0[negative_indices]
        weights1=weights1[positive_indices]
        weights_full=np.concatenate((weights0,weights1))
        weights_full=weights_full[ind[0:]]
    else:
        weights_full=np.ones(len(idxs_full))
    return(xtr_batch, ytr_batch, data_batch, weights_full, idxs_full)

def get_batch_val(neg_positive_ratio):
    random.seed(3)
    np.random.seed(3)
    noTr_positives=48200 #number positive examples in train set
    noTr_negatives_start=noTr_positives*neg_positive_ratio
    noTr_negatives=noTr_positives*neg_positive_ratio #no. negatives examples in train set
    noTr = noTr_positives + (noTr_negatives) #total size of train set
    #only sample from first 90% of dataset; shuffle first
    data1=[]
    f=open('./Datasets/icsd_full_data_unique_no_frac_no_penta_2020.txt')
    i=0
    for line in f:
        if(i>noTr_positives and i<noTr_positives*1.05):
            data1.append(line.replace('\n',''))
        i+=1
    f.close()
    data0=[]
    f=open('./Datasets/full_unsynthesized_examples.txt')
    i=0
    for line in f:
        if(i>noTr_negatives_start and i<noTr_negatives_start + (noTr_negatives*0.05)):
            data0.append(line.replace('\n',''))
        i+=1
    f.close() 
    #shuffle the positive and negative examples with themselves
    negative_indices=list(range(0,len(data0)))
    random.shuffle(negative_indices)
    positive_indices=list(range(0,len(data1)))
    random.shuffle(positive_indices)
    data0=np.array(data0)
    data1=np.array(data1)
    data0=data0[negative_indices]
    data1=data1[positive_indices]
    featmat0=get_features(data0)
    featmat1=get_features(data1)
    #get labels
    labs = np.zeros((len(data0) + len(data1),1))
    for ind,ent in enumerate(data1):
        labs[ind,0] = 1
    unique, counts = np.unique(labs, return_counts=True)
    ind0 = np.where(labs==0)[0] #indices of label=0
    ind1 = np.where(labs==1)[0] #indices of label=1
    #print(len(ind0),len(ind1))
    #combine positives and negatives and shuffle
    featmat3 = np.concatenate((featmat0,featmat1)) #set legths of labels 0 and 1 to be the same in the new feature matrix featmat3
    datasorted = np.concatenate((data0,data1)) #data ordered the same as featmat3
    labs3 = np.concatenate((labs[ind0], labs[ind1]), axis=0) #labels ordered the same as featmat3
    noS = len(featmat3)
    ind = list(range(0,noS)) #training set index
    random.shuffle(ind) #shuffle training set index
    labs3 = np.column_stack((labs3,np.abs(labs3-1)))
    xtr_batch = featmat3[ind[0:],:]
    ytr_batch = labs3[ind[0:],:]
    data_batch=datasorted[ind[0:]]
    return(xtr_batch, ytr_batch, data_batch)

def get_recall(preds,actual_values,precision=0.50):
    difference=100 #variable for how close the precision is to the desired value
    for cutoff_value in np.linspace(1,0,100):
        TP, FP, TN, FN=perf_measure(np.array(actual_values)[:,0],np.array(preds)[:,0], cutoff=cutoff_value)
        if(TP>0):
            cutoff_precision=TP/(TP+FP)
            difference=cutoff_precision-precision
            if(difference<0):
                return(TP/(TP+FN), cutoff_value)

def get_batch_test(neg_positive_ratio, seed=3):
    random.seed(seed)
    np.random.seed(seed)
    noTr_positives=48200 #number positive examples in train set
    noTr_negatives_start=noTr_positives*neg_positive_ratio
    noTr_negatives=noTr_positives*neg_positive_ratio #no. negatives examples in train set
    noTr = noTr_positives + (noTr_negatives) #total size of train set
    #only sample from first 90% of dataset ; shuffle first
    data1=[]
    f=open('./Datasets/icsd_full_data_unique_no_frac_no_penta_2020.txt')
    i=0
    for line in f:
        if(i>noTr_positives*1.05 and i<noTr_positives*1.10):
            data1.append(line.replace('\n',''))
        i+=1
    f.close()
    data0=[]
    f=open('./Datasets/full_unsynthesized_examples.txt')
    i=0
    for line in f:
        if(i>noTr_negatives_start*1.05 and i<noTr_negatives_start*1.05 + (noTr_negatives*0.05)):
            data0.append(line.replace('\n',''))
        i+=1
    f.close()
    #shuffle the positive and negative examples with themselves
    negative_indices=list(range(0,len(data0)))
    random.shuffle(negative_indices)
    positive_indices=list(range(0,len(data1)))
    random.shuffle(positive_indices)
    data0=np.array(data0)
    data1=np.array(data1)
    data0=data0[negative_indices]
    data1=data1[positive_indices]
    featmat0=get_features(data0)
    featmat1=get_features(data1)
    #get labels
    labs = np.zeros((len(data0) + len(data1),1))
    for ind,ent in enumerate(data1):
        labs[ind,0] = 1
    unique, counts = np.unique(labs, return_counts=True)
    ind0 = np.where(labs==0)[0] #indices of label=0
    ind1 = np.where(labs==1)[0] #indices of label=1
    #print(len(ind0),len(ind1))
    #combine positives and negatives and shuffle
    featmat3 = np.concatenate((featmat0,featmat1)) #set legths of labels 0 and 1 to be the same in the new feature matrix featmat3
    datasorted = np.concatenate((data0,data1)) #data ordered the same as featmat3
    labs3 = np.concatenate((labs[ind0], labs[ind1]), axis=0) #labels ordered the same as featmat3
    noS = len(featmat3)
    ind = list(range(0,noS)) #training set index
    random.shuffle(ind) #shuffle training set index
    #indB = list(range(0,noTr_subset)) #used later for batch
    labs3 = np.column_stack((labs3,np.abs(labs3-1)))
    xtr_batch = featmat3[ind[0:],:]
    ytr_batch = labs3[ind[0:],:]
    data_batch=datasorted[ind[0:]]
    return(xtr_batch, ytr_batch, data_batch)

neg_pos_ratio=25
weight_for_0 = (1 + neg_pos_ratio) / (2*neg_pos_ratio)
weight_for_1 = (1 + neg_pos_ratio) / (2*1)
def perf_measure(y_actual, y_hat, cutoff=0.5):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==1 and y_hat[i]>cutoff:
            TP += 1
        if y_hat[i]>cutoff and y_actual[i]==0:
            FP += 1
        if y_actual[i]==0 and y_hat[i]<cutoff:
            TN += 1
        if y_hat[i]<cutoff and y_actual[i]==1:
            FN += 1

    return(TP, FP, TN, FN)

#charge balancing over time
#full_ox_dict = {'H': [-1, 0, 1], 'He': [0], 'Li': [0, 1], 'Be': [0, 1, 2], 'B': [-5, -1, 0, 1, 2, 3], 'C': [-4, -3, -2, -1, 0, 1, 2, 3, 4], 'N': [-3, -2, -1, 0, 1, 2, 3, 4, 5], 'O': [-2, -1, 0, 1, 2], 'F': [-1, 0], 'Ne': [0], 'Na': [-1, 0, 1], 'Mg': [0, 1, 2], 'Al': [-2, -1, 0, 1, 2, 3], 'Si': [-4, -3, -2, -1, 0, 1, 2, 3, 4], 'P': [-3, -2, -1, 0, 1, 2, 3, 4, 5], 'S': [-2, -1, 0, 1, 2, 3, 4, 5, 6], 'Cl': [-1, 0, 1, 2, 3, 4, 5, 6, 7], 'Ar': [0], 'K': [-1, 0, 1], 'Ca': [0, 1, 2], 'Sc': [0, 1, 2, 3], 'Ti': [-2, -1, 0, 1, 2, 3, 4], 'V': [-3, -1, 0, 1, 2, 3, 4, 5], 'Cr': [-4, -2, -1, 0, 1, 2, 3, 4, 5, 6], 'Mn': [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7], 'Fe': [-4, -2, -1, 0, 1, 2, 3, 4, 5, 6], 'Co': [-3, -1, 0, 1, 2, 3, 4, 5], 'Ni': [-2, -1, 0, 1, 2, 3, 4], 'Cu': [-2, 0, 1, 2, 3, 4], 'Zn': [-2, 0, 1, 2], 'Ga': [-5, -4, -2, -1, 0, 1, 2, 3], 'Ge': [-4, -3, -2, -1, 0, 1, 2, 3, 4], 'As': [-3, -2, -1, 0, 1, 2, 3, 4, 5], 'Se': [-2, -1, 0, 1, 2, 3, 4, 5, 6], 'Br': [-1, 0, 1, 3, 4, 5, 7], 'Kr': [0, 2], 'Rb': [-1, 0, 1], 'Sr': [0, 1, 2], 'Y': [0, 1, 2, 3], 'Zr': [-2, 0, 1, 2, 3, 4], 'Nb': [-3, -1, 0, 1, 2, 3, 4, 5], 'Mo': [-4, -2, -1, 0, 1, 2, 3, 4, 5, 6], 'Tc': [-3, -1, 0, 1, 2, 3, 4, 5, 6, 7], 'Ru': [-4, -2, 0, 1, 2, 3, 4, 5, 6, 7, 8], 'Rh': [-3, -1, 0, 1, 2, 3, 4, 5, 6], 'Pd': [0, 1, 2, 3, 4, 5, 6], 'Ag': [-2, -1, 0, 1, 2, 3, 4], 'Cd': [-2, 0, 1, 2], 'In': [-5, -2, -1, 0, 1, 2, 3], 'Sn': [-4, -3, -2, -1, 0, 1, 2, 3, 4], 'Sb': [-3, -2, -1, 0, 1, 2, 3, 4, 5], 'Te': [-2, -1, 0, 1, 2, 3, 4, 5, 6], 'I': [-1, 0, 1, 3, 4, 5, 6, 7], 'Xe': [0, 2, 4, 6, 8], 'Cs': [-1, 0, 1], 'Ba': [0, 1, 2], 'Hf': [-2, 0, 1, 2, 3, 4], 'Ta': [-3, -1, 0, 1, 2, 3, 4, 5], 'W': [-4, -2, -1, 0, 1, 2, 3, 4, 5, 6], 'Re': [-3, -1, 0, 1, 2, 3, 4, 5, 6, 7], 'Os': [-4, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], 'Ir': [-3, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'Pt': [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6], 'Au': [-3, -2, -1, 0, 1, 2, 3, 5], 'Hg': [-2, 0, 1, 2], 'Tl': [-5, -2, -1, 0, 1, 2, 3], 'Pb': [-4, -2, -1, 0, 1, 2, 3, 4], 'Bi': [-3, -2, -1, 0, 1, 2, 3, 4, 5], 'Po': [-2, 0, 2, 4, 5, 6], 'At': [-1, 0, 1, 3, 5, 7], 'Rn': [0, 2, 6], 'Fr': [0, 1], 'Ra': [0, 2], 'Rf': [0, 4], 'Db': [0, 5], 'Sg': [0, 6], 'Bh': [0, 7], 'Hs': [0, 8], 'Mt': [0], 'Ds': [0], 'Rg': [0], 'Cn': [0], 'Nh': [0], 'Fl': [0], 'Mc': [0], 'Lv': [0], 'Ts': [0], 'Og': [0], 'La': [0, 1, 2, 3], 'Ce': [0, 2, 3, 4], 'Pr': [0, 2, 3, 4], 'Nd': [0, 2, 3, 4], 'Pm': [0, 2, 3], 'Sm': [0, 2, 3], 'Eu': [0, 2, 3], 'Gd': [0, 1, 2, 3], 'Tb': [0, 1, 2, 3, 4], 'Dy': [0, 2, 3, 4], 'Ho': [0, 2, 3], 'Er': [0, 2, 3], 'Tm': [0, 2, 3], 'Yb': [0, 2, 3], 'Lu': [0, 2, 3], 'Ac': [0, 2, 3], 'Th': [0, 1, 2, 3, 4], 'Pa': [0, 2, 3, 4, 5], 'U': [0, 1, 2, 3, 4, 5, 6], 'Np': [0, 2, 3, 4, 5, 6, 7], 'Pu': [0, 1, 2, 3, 4, 5, 6, 7, 8], 'Am': [0, 2, 3, 4, 5, 6, 7, 8], 'Cm': [0, 2, 3, 4, 6], 'Bk': [0, 2, 3, 4], 'Cf': [0, 2, 3, 4], 'Es': [0, 2, 3, 4], 'Fm': [0, 2, 3], 'Md': [0, 2, 3], 'No': [0, 2, 3], 'Lr': [0, 3]}
#common_ox_dict = {'H': [-1, 0, 1], 'He': [0], 'Li': [0, 1], 'Be': [0, 2], 'B': [0, 3], 'C': [-4, -3, -2, -1, 0, 1, 2, 3, 4], 'N': [-3, 0, 3, 5], 'O': [-2, 0], 'F': [-1, 0], 'Ne': [0], 'Na': [0, 1], 'Mg': [0, 2], 'Al': [0, 3], 'Si': [-4, 0, 4], 'P': [-3, 0, 3, 5], 'S': [-2, 0, 2, 4, 6], 'Cl': [-1, 0, 1, 3, 5, 7], 'Ar': [0], 'K': [0, 1], 'Ca': [0, 2], 'Sc': [0, 3], 'Ti': [0, 4], 'V': [0, 5], 'Cr': [0, 3, 6], 'Mn': [0, 2, 4, 7], 'Fe': [0, 2, 3, 6], 'Co': [0, 2, 3], 'Ni': [0, 2], 'Cu': [0, 2], 'Zn': [0, 2], 'Ga': [0, 3], 'Ge': [-4, 0, 2, 4], 'As': [-3, 0, 3, 5], 'Se': [-2, 0, 2, 4, 6], 'Br': [-1, 0, 1, 3, 5, 7], 'Kr': [0, 2], 'Rb': [0, 1], 'Sr': [0, 2], 'Y': [0, 3], 'Zr': [0, 4], 'Nb': [0, 5], 'Mo': [0, 4, 6], 'Tc': [0, 4, 7], 'Ru': [0, 2, 3, 4], 'Rh': [0, 3], 'Pd': [0, 2, 4], 'Ag': [0, 1], 'Cd': [0, 2], 'In': [0, 3], 'Sn': [-4, 0, 2, 4], 'Sb': [-3, 0, 3, 5], 'Te': [-2, 0, 2, 4, 6], 'I': [-1, 0, 1, 3, 5, 7], 'Xe': [0, 2, 4, 6], 'Cs': [0, 1], 'Ba': [0, 2], 'Hf': [0, 4], 'Ta': [0, 5], 'W': [0, 4, 6], 'Re': [0, 4], 'Os': [0, 4], 'Ir': [0, 3, 4], 'Pt': [0, 2, 4], 'Au': [0, 3], 'Hg': [0, 1, 2], 'Tl': [0, 1, 3], 'Pb': [0, 2, 4], 'Bi': [0, 3], 'Po': [-2, 0, 2, 4], 'At': [-1, 0, 1], 'Rn': [0, 2], 'Fr': [0, 1], 'Ra': [0, 2], 'Rf': [0, 4], 'Db': [0, 5], 'Sg': [0, 6], 'Bh': [0, 7], 'Hs': [0, 8], 'Mt': [0], 'Ds': [0], 'Rg': [0], 'Cn': [0], 'Nh': [0], 'Fl': [0], 'Mc': [0], 'Lv': [0], 'Ts': [0], 'Og': [0], 'La': [0, 3], 'Ce': [0, 3, 4], 'Pr': [0, 3], 'Nd': [0, 3], 'Pm': [0, 3], 'Sm': [0, 3], 'Eu': [0, 2, 3], 'Gd': [0, 3], 'Tb': [0, 3], 'Dy': [0, 3], 'Ho': [0, 3], 'Er': [0, 3], 'Tm': [0, 3], 'Yb': [0, 3], 'Lu': [0, 3], 'Ac': [0, 3], 'Th': [0, 4], 'Pa': [0, 5], 'U': [0, 4, 6], 'Np': [0, 5], 'Pu': [0, 4], 'Am': [0, 3], 'Cm': [0, 3], 'Bk': [0, 3], 'Cf': [0, 3], 'Es': [0, 3], 'Fm': [0, 3], 'Md': [0, 3], 'No': [0, 2], 'Lr': [0, 3]}
mg_common_ox_dict = {}
mg_full_ox_dict = {}
species_in_use = ['Ac', 'Ag', 'Al', 'As', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Br', 'C', 'Ca', 'Cd', 'Ce', 'Cl', 'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Er', 'Eu', 'F', 'Fe', 'Ga', 'Gd', 'Ge', 'H', 'Hf', 'Hg', 'Ho', 'I', 'In', 'Ir', 'K', 'La', 'Li', 'Lu', 'Mg', 'Mn', 'Mo', 'N', 'Na', 'Nb', 'Nd', 'Ni', 'Np', 'O', 'Os', 'P', 'Pa', 'Pb', 'Pd', 'Pr', 'Pt', 'Pu', 'Rb', 'Re', 'Rh', 'Ru', 'S', 'Sb', 'Sc', 'Se', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Te', 'Th', 'Ti', 'Tl', 'Tm', 'U', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr']
for specie in species_in_use:
    mg_common_ox_dict[specie] = list(mg.core.periodic_table.Specie(specie).common_oxidation_states)
for specie in species_in_use:
    mg_full_ox_dict[specie] = list(mg.core.periodic_table.Specie(specie).oxidation_states)
full_ox_dict = mg_full_ox_dict
common_ox_dict = mg_common_ox_dict
full_ox_dict = {x: full_ox_dict[x] for x in full_ox_dict if x in species_in_use}
common_ox_dict = {x: common_ox_dict[x] for x in common_ox_dict if x in species_in_use}
element_names_array=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl', 'Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu']

def load_data(data, is_charge_balanced, max_atoms=5, max_coefficient=100000):
    #takes input file (icsd_full_properties_no_frac_charges) and processes the data and applies some filters
    output_array=[]
    coeff_array=np.zeros((10000,1))
    element_names_array=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl', 'Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu']

    for i in range(len(data)):
        try:
            comp=mg.core.composition.Composition(data[i,1])
        except:
            continue #bad formula

        if(len(mg.core.composition.Composition(data[i,1]))==1):
            continue

        truth_array=[]
        for element_name in mg.core.composition.Composition(data[i,1]).as_dict().keys():
            if(element_name not in element_names_array):
                truth_array.append('False')
        if('False' in truth_array):
            continue

        if(is_charge_balanced):
            if('True' in data[i][8]):
                if(len(mg.core.composition.Composition(data[i,1]))<max_atoms):
                    values=mg.core.composition.Composition(data[i,1]).as_dict().values()
                    for value in values:
                        coeff_array[int(value)]=coeff_array[int(value)]+1
                    large_values=[x for x in values if x>max_coefficient]
                    if(len(large_values)==0):
                        output_array.append(mg.core.composition.Composition(data[i,1]).alphabetical_formula.replace(' ', ''))
        else:
            output_array.append(mg.core.composition.Composition(data[i,1]).alphabetical_formula.replace(' ', ''))
    return(np.unique(output_array))

def get_features(data0):
    p = re.compile('[A-Z][a-z]?\d*\.?\d*')
    p3 = re.compile('[A-Z][a-z]?')
    p5 = re.compile('\d+\.?\d+|\d+')
    data0_ratio=[]
    for i in data0:
        x = i
        p2 = p.findall(x)
        temp1,temp2 = [], []
        for x in p2:
            temp1.append(Element[p3.findall(x)[0]].number)
            kkk = p5.findall(x)
            if len(kkk)<1:
                temp2.append(1)
            else:
                temp2.append(kkk[0])
        data0_ratio.append([temp1,list(map(float,temp2))])

    I = 94
    featmat0 = np.zeros((len(data0_ratio),I))
    # featmat: n-hot vectors with fractions
    for idx,ent in enumerate(data0_ratio):
        for idy,at in enumerate(ent[0]):
            featmat0[idx,at-1] = ent[1][idy]/sum(ent[1])
    return(featmat0)


def make_negative_data(num_examples, max_atoms=5, max_coefficient=11, seed=3, weighted=False):
    output_array=[]
    element_names_array=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl', 'Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu']
    element_sum=np.loadtxt('./Datasets/icsd_unique_element_proportions.txt')
    random.seed(seed)
    np.random.seed(seed)

    while len(output_array) < num_examples:
        if(len(output_array)%10000==0):
            a=1
        num_atoms=np.random.randint(2,max_atoms,1)[0] #number of atom types (binary, tern,  quat)
        coeffs=np.random.randint(1,max_coefficient,num_atoms) #coeffs for each atom
        if(weighted):
            atomic_numbers=np.random.choice(94, num_atoms, p=np.reshape(element_sum, [94]))+1 #add one to give between 1,95
        else:
            atomic_numbers=np.random.randint(1,95,num_atoms)  #goes up to atomic number 94

        output=''
        for i in range(num_atoms):
            output+=element_names_array[atomic_numbers[i]-1]
            output+=str(coeffs[i])
        if(mg.core.composition.Composition(output).alphabetical_formula.replace(' ', '') not in output_array):
            output_array.append(mg.core.composition.Composition(output).alphabetical_formula.replace(' ', ''))
    return(output_array)

def get_coeffs(data):
    coeffs = []
    for entry in data:
        coeffs.append(list(mg.core.Composition(entry).as_dict().values()))
    return coeffs

def get_max_coeff(coeffs):
    max_entry = 0
    for entry in coeffs:
        if max(entry) > max_entry:
            max_entry = max(entry)
    return max_entry
def filter_data(data, max_coeff=100, species_counts=[1,2,3,4,5], mandatory_species=[]):
    filtered_data = []
    for entry in data:
        # check coefficients
        try:
            if get_max_coeff(get_coeffs([entry])) > max_coeff:
                continue
        except:
            continue

        # check length
        if len(mg.core.Composition(entry).elements) not in species_counts:
            continue

        # check that all species are from species_in use
        if False in [species in species_in_use for species in list(mg.core.Composition(entry).as_dict().keys())]:
            continue

        # mandatory species
        mandatory_count = len(mandatory_species)
        mandatory_present = 0
        if mandatory_count > 0:
            for species in entry:
                if species in mandatory_species:
                    mandatory_present += 1
            if mandatory_present != mandatory_count:
                continue

        filtered_data.append(entry)

    return filtered_data

def SynthNN_best_model(x_input,y_input,data_input):
    #predicts the synthesizability using the best performing SynthNN model
    num_positive=41599
    tf.compat.v1.disable_eager_execution()
    M =30
    DIR='All_hyperparam_training/' +str(M) + 'M/20/'
    name='performance_matrix_TL_v3_30M_14112.txt'

    hyperparams=[name[-9],name[-8],name[-7],name[-6],name[-5]]
    hyperparams=np.array(hyperparams, dtype=int)
    no_h1=[30,40,50,60,80][hyperparams[1]]
    no_h2=[30,40,50,60,80][hyperparams[2]]
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_input.shape[1]])
    y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
    W1=tf.compat.v1.placeholder(tf.float32, shape=[x_input.shape[1],M]) #
    F1 = tf.compat.v1.placeholder(tf.float32, shape=[M,no_h1])
    F2 = tf.compat.v1.placeholder(tf.float32, shape=[no_h1,no_h2])
    F3 = tf.compat.v1.placeholder(tf.float32, shape=[no_h2,2])
    b1 = tf.compat.v1.placeholder(tf.float32, shape=[no_h1])
    b2 = tf.compat.v1.placeholder(tf.float32, shape=[no_h2])
    b3 = tf.compat.v1.placeholder(tf.float32, shape=[2])
    sess = tf.compat.v1.InteractiveSession()
    z0_raw = tf.multiply(tf.expand_dims(x,2),tf.expand_dims(W1,0)) #(ntr, I, M)
    tempmean,var = tf.nn.moments(x=z0_raw,axes=[1])
    z0 = tf.concat([tf.reduce_sum(input_tensor=z0_raw,axis=1)],1) #(ntr, M)
    z1 = tf.add(tf.matmul(z0,F1),b1) #(ntr, no_h1)
    a1 = tf.tanh(z1) #(ntr, no_h1)
    z2= tf.add(tf.matmul(a1,F2),b2) #(ntr,no_h1)
    a2= tf.tanh(z2) #(ntr, no_h1)
    z3 = tf.add(tf.matmul(a2,F3),b3) #(ntr, 2)
    a3 = tf.nn.softmax(z3) #(ntr, 2)
    clipped_y = tf.clip_by_value(a3, 1e-10, 1.0)
    cross_entropy = -tf.reduce_sum(input_tensor=y_*tf.math.log(clipped_y)*np.array([weight_for_1,weight_for_0]))
    correct_prediction = tf.equal(tf.argmax(input=a3,axis=1), tf.argmax(input=y_,axis=1))
    accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))
    sess.run(tf.compat.v1.initialize_all_variables())
    model_name=str(M) + 'M_synth_v3_semi' + str(hyperparams[0]) + str(hyperparams[1])+ str(hyperparams[2])+ str(hyperparams[3])+ str(hyperparams[4])  +'.txt'
    directory=DIR + '/'
    W1_loaded=np.loadtxt(directory + 'W1_' + model_name) 
    F1_loaded=np.loadtxt(directory + 'F1_' + model_name) 
    F2_loaded=np.loadtxt(directory + 'F2_' + model_name) 
    F3_loaded=np.loadtxt(directory + 'F3_' + model_name) 
    F3_loaded=np.reshape(F3_loaded, [no_h2,2])
    b1_loaded=np.loadtxt(directory + 'b1_' + model_name) 
    b2_loaded=np.loadtxt(directory + 'b2_' + model_name) 
    b3_loaded=np.loadtxt(directory + 'b3_' + model_name)
    b3_loaded=np.reshape(b3_loaded, [2])
    preds=a3.eval(feed_dict={x: x_input, y_: y_input , W1:W1_loaded, F1:F1_loaded, F2:F2_loaded, F3:F3_loaded, b1:b1_loaded, b2:b2_loaded, b3:b3_loaded})
    #te_accuracy=accuracy.eval(feed_dict={x: x_input, y_: y_input , W1:W1_loaded, F1:F1_loaded, F2:F2_loaded, F3:F3_loaded, b1:b1_loaded, b2:b2_loaded, b3:b3_loaded})
    sess.close()
    return(preds)

def get_model_preds(M,nsynth_train,hyperparameter_string,formulas):
    #given input hyperparameters, M, nsynth_train and hyperparameter string, returns the predicted synthesizability
    #formulas: array of input formulas string, i.e. ['NaCl','K2O']
    tf.compat.v1.disable_eager_execution()
    DIR='All_hyperparam_training/' +str(M) + 'M/' + str(nsynth_train) +'/'
    x_data=get_features(formulas)
    y_data=np.zeros((len(formulas),2)) #make fake y_data
    for name in os.listdir(DIR):
        if(name.endswith('.txt')):
            if(name.startswith('performance_matrix_TL_v3_' + str(M) + 'M_' + hyperparameter_string)):
                hyperparams=[name[-9],name[-8],name[-7],name[-6],name[-5]]
                hyperparams=np.array(hyperparams, dtype=int)
                no_h1=[30,40,50,60,80][hyperparams[1]]
                no_h2=[30,40,50,60,80][hyperparams[2]]
                x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_data.shape[1]])
                y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
                W1=tf.compat.v1.placeholder(tf.float32, shape=[x_data.shape[1],M]) #if loading in weights for TL
                F1 = tf.compat.v1.placeholder(tf.float32, shape=[M,no_h1])
                F2 = tf.compat.v1.placeholder(tf.float32, shape=[no_h1,no_h2])
                F3 = tf.compat.v1.placeholder(tf.float32, shape=[no_h2,2])
                b1 = tf.compat.v1.placeholder(tf.float32, shape=[no_h1])
                b2 = tf.compat.v1.placeholder(tf.float32, shape=[no_h2])
                b3 = tf.compat.v1.placeholder(tf.float32, shape=[2])
                sess = tf.compat.v1.InteractiveSession()
                sess.run(tf.compat.v1.initialize_all_variables())
                z0_raw = tf.multiply(tf.expand_dims(x,2),tf.expand_dims(W1,0)) #(ntr, I, M)
                tempmean,var = tf.nn.moments(x=z0_raw,axes=[1])
                z0 = tf.concat([tf.reduce_sum(input_tensor=z0_raw,axis=1)],1) #(ntr, M)
                z1 = tf.add(tf.matmul(z0,F1),b1) #(ntr, no_h1)
                a1 = tf.tanh(z1) #(ntr, no_h1)
                z2= tf.add(tf.matmul(a1,F2),b2) #(ntr,no_h1)
                a2= tf.tanh(z2) #(ntr, no_h1)
                z3 = tf.add(tf.matmul(a2,F3),b3) #(ntr, 2)
                a3 = tf.nn.softmax(z3) #(ntr, 2)
                clipped_y = tf.clip_by_value(a3, 1e-10, 1.0)
                cross_entropy = -tf.reduce_sum(input_tensor=y_*tf.math.log(clipped_y)*np.array([weight_for_1,weight_for_0]))
                correct_prediction = tf.equal(tf.argmax(input=a3,axis=1), tf.argmax(input=y_,axis=1))
                accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))
                sess.run(tf.compat.v1.initialize_all_variables())
                model_name=str(M) + 'M_synth_v3_semi' + str(hyperparams[0]) + str(hyperparams[1])+ str(hyperparams[2])+ str(hyperparams[3])+ str(hyperparams[4])  +'.txt'
                directory=DIR + '/'
                W1_loaded=np.loadtxt(directory + 'W1_' + model_name) #load weights if doing TL HERE!!
                F1_loaded=np.loadtxt(directory + 'F1_' + model_name) #load weights if doing TL HERE!!
                F2_loaded=np.loadtxt(directory + 'F2_' + model_name) #load weights if doing TL HERE!!
                F3_loaded=np.loadtxt(directory + 'F3_' + model_name) #load weights if doing TL HERE!!
                F3_loaded=np.reshape(F3_loaded, [no_h2,2])
                b1_loaded=np.loadtxt(directory + 'b1_' + model_name) #load weights if doing TL HERE!!
                b2_loaded=np.loadtxt(directory + 'b2_' + model_name) #load weights if doing TL HERE!!
                b3_loaded=np.loadtxt(directory + 'b3_' + model_name) #load weights if doing TL HERE!!
                b3_loaded=np.reshape(b3_loaded, [2])
                sess.run(tf.compat.v1.initialize_all_variables())
                preds=a3.eval(feed_dict={x: x_data, y_: y_data , W1:W1_loaded, F1:F1_loaded, F2:F2_loaded, F3:F3_loaded, b1:b1_loaded, b2:b2_loaded, b3:b3_loaded})
                sess.close()
                return(preds)
            
def get_decade_model_preds(decade,formulas):
    #given input hyperparameters, M, nsynth_train and hyperparameter string, returns the predicted synthesizability
    #formulas: array of input formulas string, i.e. ['NaCl','K2O']
    nsynth_train=20
    M=30
    hyperparameter_string='14112'
    DIR='All_hyperparam_training/'+decade+ '_best/'
    x_data=get_features(formulas)
    y_data=np.zeros((len(formulas),2)) #make fake y_data
    name='performance_matrix_TL_v3_' + str(M) + 'M_' + hyperparameter_string + '.txt'
    hyperparams=[name[-9],name[-8],name[-7],name[-6],name[-5]]
    hyperparams=np.array(hyperparams, dtype=int)
    no_h1=[30,40,50,60,80][hyperparams[1]]
    no_h2=[30,40,50,60,80][hyperparams[2]]
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, x_data.shape[1]])
    y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
    W1=tf.compat.v1.placeholder(tf.float32, shape=[x_data.shape[1],M]) #if loading in weights for TL
    F1 = tf.compat.v1.placeholder(tf.float32, shape=[M,no_h1])
    F2 = tf.compat.v1.placeholder(tf.float32, shape=[no_h1,no_h2])
    F3 = tf.compat.v1.placeholder(tf.float32, shape=[no_h2,2])
    b1 = tf.compat.v1.placeholder(tf.float32, shape=[no_h1])
    b2 = tf.compat.v1.placeholder(tf.float32, shape=[no_h2])
    b3 = tf.compat.v1.placeholder(tf.float32, shape=[2])
    sess = tf.compat.v1.InteractiveSession()
    z0_raw = tf.multiply(tf.expand_dims(x,2),tf.expand_dims(W1,0)) #(ntr, I, M)
    tempmean,var = tf.nn.moments(x=z0_raw,axes=[1])
    z0 = tf.concat([tf.reduce_sum(input_tensor=z0_raw,axis=1)],1) #(ntr, M)
    z1 = tf.add(tf.matmul(z0,F1),b1) #(ntr, no_h1)
    a1 = tf.tanh(z1) #(ntr, no_h1)
    z2= tf.add(tf.matmul(a1,F2),b2) #(ntr,no_h1)
    a2= tf.tanh(z2) #(ntr, no_h1)
    z3 = tf.add(tf.matmul(a2,F3),b3) #(ntr, 2)
    a3 = tf.nn.softmax(z3) #(ntr, 2)
    clipped_y = tf.clip_by_value(a3, 1e-10, 1.0)
    cross_entropy = -tf.reduce_sum(input_tensor=y_*tf.math.log(clipped_y)*np.array([weight_for_1,weight_for_0]))
    correct_prediction = tf.equal(tf.argmax(input=a3,axis=1), tf.argmax(input=y_,axis=1))
    accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))
    model_name=str(M) + 'M_synth_v3_semi' + str(hyperparams[0]) + str(hyperparams[1])+ str(hyperparams[2])+ str(hyperparams[3])+ str(hyperparams[4])  +'.txt'
    W1_loaded=np.loadtxt(DIR + 'W1_' + model_name) 
    F1_loaded=np.loadtxt(DIR + 'F1_' + model_name)
    F2_loaded=np.loadtxt(DIR + 'F2_' + model_name) 
    F3_loaded=np.loadtxt(DIR + 'F3_' + model_name) 
    F3_loaded=np.reshape(F3_loaded, [no_h2,2])
    b1_loaded=np.loadtxt(DIR + 'b1_' + model_name) 
    b2_loaded=np.loadtxt(DIR + 'b2_' + model_name) 
    b3_loaded=np.loadtxt(DIR + 'b3_' + model_name) 
    b3_loaded=np.reshape(b3_loaded, [2])
    sess.run(tf.compat.v1.initialize_all_variables())
    preds=a3.eval(feed_dict={x: x_data, y_: y_data , W1:W1_loaded, F1:F1_loaded, F2:F2_loaded, F3:F3_loaded, b1:b1_loaded, b2:b2_loaded, b3:b3_loaded})
    sess.close()
    return(preds)
                     
def get_block_performance(data,yvalues):
    #only used in making Figure 3c
    #inputs:
    #data: array of formulas to make predictions on
    #yvalues: ground truth labels of the formulas in data
    #returns: the F1-scores of the s_block,p_block,d_block and f_block containing formulas

    s_block_performance=0
    p_block_performance=0
    d_block_performance=0
    f_block_performance=0
    synthNN_TP_block_pred_dict = {"s": 0,"p": 0,"d": 0,"f":0}
    synthNN_FP_block_pred_dict = {"s": 0,"p": 0,"d": 0,"f":0}
    synthNN_TN_block_pred_dict = {"s": 0,"p": 0,"d": 0,"f":0}
    synthNN_FN_block_pred_dict = {"s": 0,"p": 0,"d": 0,"f":0}
    x_data=get_features(data)
    dummy_y_values=np.zeros((len(data),2))
    synthNN_quiz_preds=SynthNN_best_model(x_data,dummy_y_values,data)[:,0]

    for i in range(len(data)):
        elements=mg.core.composition.Composition(data[i]).as_dict().keys()
        blocks_in_formula=np.unique([mg.core.periodic_table.Element(element).block for element in elements])
        for element_block_name in blocks_in_formula:
            if(yvalues[i]==1):
                if(synthNN_quiz_preds[i]>0.5):
                    synthNN_TP_block_pred_dict[element_block_name]=synthNN_TP_block_pred_dict[element_block_name]+1
                else:
                    synthNN_FN_block_pred_dict[element_block_name]=synthNN_FN_block_pred_dict[element_block_name]+1
            else:
                if(synthNN_quiz_preds[i]<0.5):
                    synthNN_TN_block_pred_dict[element_block_name]=synthNN_TN_block_pred_dict[element_block_name]+1
                else:
                    synthNN_FP_block_pred_dict[element_block_name]=synthNN_FP_block_pred_dict[element_block_name]+1

    #throw exception in case of no instances of that block
    try:
        s_block_performance=(synthNN_TP_block_pred_dict['s']/(synthNN_TP_block_pred_dict['s']+(0.5*(synthNN_FP_block_pred_dict['s']+synthNN_FN_block_pred_dict['s']))))
    except ZeroDivisionError:
        pass
    try:
        p_block_performance=(synthNN_TP_block_pred_dict['p']/(synthNN_TP_block_pred_dict['p']+(0.5*(synthNN_FP_block_pred_dict['p']+synthNN_FN_block_pred_dict['p']))))
    except ZeroDivisionError:
        pass
    try:
        d_block_performance=(synthNN_TP_block_pred_dict['d']/(synthNN_TP_block_pred_dict['d']+(0.5*(synthNN_FP_block_pred_dict['d']+synthNN_FN_block_pred_dict['d']))))
    except ZeroDivisionError:
        pass
    try:
        f_block_performance=(synthNN_TP_block_pred_dict['f']/(synthNN_TP_block_pred_dict['f']+(0.5*(synthNN_FP_block_pred_dict['f']+synthNN_FN_block_pred_dict['f']))))
    except ZeroDivisionError:
        pass
    return(s_block_performance,p_block_performance,d_block_performance,f_block_performance)

def get_element_type_color(symbol):
    #given input symbol, return color to use in plot for Figure 5b
    c=''

    if(mg.core.periodic_table.Element(symbol).is_transition_metal):
        return('purple')
    elif(symbol=='O'):
        return('red')
    elif(mg.core.periodic_table.Element(symbol).is_post_transition_metal):
        c='orange'
    elif(mg.core.periodic_table.Element(symbol).is_rare_earth_metal):
        c='green'
    elif(symbol=='As' or symbol=='N' or symbol=='P'):
        return('blue')
    elif(mg.core.periodic_table.Element(symbol).is_metalloid):
        c='cyan'
    elif(mg.core.periodic_table.Element(symbol).is_halogen):
        c='pink'
    elif(mg.core.periodic_table.Element(symbol).is_chalcogen):
        c='yellow'
    elif(symbol=='C'):
        c='gray'
    elif(mg.core.periodic_table.Element(symbol).is_alkali):
        c='yellow'
    else:
        c='white'
    return(c)

def get_batch_val_2000(neg_positive_ratio):
    random.seed(3)
    np.random.seed(3)
    noTr_positives=30400 #number positive examples in train set
    noTr_negatives=noTr_positives*neg_positive_ratio #no. negatives examples in train set
    noTr = noTr_positives + (noTr_negatives) #total size of train set

    #only sample from first 90% of dataset; shuffle first
    data1=[]
    f=open('All_hyperparam_training/icsd_full_data_unique_no_frac_2000_nopenta.txt')
    i=0
    for line in f:
        if(i>noTr_positives and i<noTr_positives*1.05):
            data1.append(line.replace('\n',''))
        i+=1
    f.close()

    data0=[]
    f=open('./Datasets/full_unsynthesized_examples.txt')
    i=0
    for line in f:
        if(i>noTr_negatives and i<noTr_negatives*1.05):
            data0.append(line.replace('\n',''))
        i+=1
    f.close()

    #shuffle the positive and negative examples with themselves
    negative_indices=list(range(0,len(data0)))
    random.shuffle(negative_indices)
    positive_indices=list(range(0,len(data1)))
    random.shuffle(positive_indices)
    data0=np.array(data0)
    data1=np.array(data1)
    data0=data0[negative_indices]
    data1=data1[positive_indices]
    featmat0=get_features(data0)
    featmat1=get_features(data1)

    #get labels
    labs = np.zeros((len(data0) + len(data1),1))
    for ind,ent in enumerate(data1):
        labs[ind,0] = 1
    unique, counts = np.unique(labs, return_counts=True)
    ind0 = np.where(labs==0)[0] #indices of label=0
    ind1 = np.where(labs==1)[0] #indices of label=1
    #print(len(ind0),len(ind1))

    #combine positives and negatives and shuffle

    featmat3 = np.concatenate((featmat0,featmat1)) #set legths of labels 0 and 1 to be the same in the new feature matrix featmat3
    datasorted = np.concatenate((data0,data1)) #data ordered the same as featmat3
    labs3 = np.concatenate((labs[ind0], labs[ind1]), axis=0) #labels ordered the same as featmat3
    noS = len(featmat3)
    ind = list(range(0,noS)) #training set index
    random.shuffle(ind) #shuffle training set index
    #indB = list(range(0,noTr_subset)) #used later for batch
    labs3 = np.column_stack((labs3,np.abs(labs3-1)))
    xtr_batch = featmat3[ind[0:],:]
    ytr_batch = labs3[ind[0:],:]
    data_batch=datasorted[ind[0:]]
    return(xtr_batch, ytr_batch, data_batch)

def get_batch_val_2010(neg_positive_ratio):

    random.seed(3)
    np.random.seed(3)
    noTr_positives=38900 #number positive examples in train set
    noTr_negatives=noTr_positives*neg_positive_ratio #no. negatives examples in train set
    noTr = noTr_positives + (noTr_negatives) #total size of train set

    #only sample from first 90% of dataset; shuffle first
    data1=[]
    f=open('All_hyperparam_training/icsd_full_data_unique_no_frac_2010_nopenta.txt')
    i=0
    for line in f:
        if(i>noTr_positives and i<noTr_positives*1.05):
            data1.append(line.replace('\n',''))
        i+=1
    f.close()

    data0=[]
    f=open('./Datasets/full_unsynthesized_examples.txt')
    i=0
    for line in f:
        if(i>noTr_negatives and i<noTr_negatives*1.05):
            data0.append(line.replace('\n',''))
        i+=1
    f.close()

    #shuffle the positive and negative examples with themselves
    negative_indices=list(range(0,len(data0)))
    random.shuffle(negative_indices)
    positive_indices=list(range(0,len(data1)))
    random.shuffle(positive_indices)
    data0=np.array(data0)
    data1=np.array(data1)
    data0=data0[negative_indices]
    data1=data1[positive_indices]
    featmat0=get_features(data0)
    featmat1=get_features(data1)

    #get labels
    labs = np.zeros((len(data0) + len(data1),1))
    for ind,ent in enumerate(data1):
        labs[ind,0] = 1
    unique, counts = np.unique(labs, return_counts=True)
    ind0 = np.where(labs==0)[0] #indices of label=0
    ind1 = np.where(labs==1)[0] #indices of label=1
    #print(len(ind0),len(ind1))

    #combine positives and negatives and shuffle
    featmat3 = np.concatenate((featmat0,featmat1)) #set legths of labels 0 and 1 to be the same in the new feature matrix featmat3
    datasorted = np.concatenate((data0,data1)) #data ordered the same as featmat3
    labs3 = np.concatenate((labs[ind0], labs[ind1]), axis=0) #labels ordered the same as featmat3
    noS = len(featmat3)
    ind = list(range(0,noS)) #training set index
    random.shuffle(ind) #shuffle training set index
    #indB = list(range(0,noTr_subset)) #used later for batch
    labs3 = np.column_stack((labs3,np.abs(labs3-1)))
    xtr_batch = featmat3[ind[0:],:]
    ytr_batch = labs3[ind[0:],:]
    data_batch=datasorted[ind[0:]]
    return(xtr_batch, ytr_batch, data_batch)

def get_batch_val_1990(neg_positive_ratio):

    random.seed(3)
    np.random.seed(3)
    noTr_positives=22600 #number positive examples in train set
    noTr_negatives=noTr_positives*neg_positive_ratio #no. negatives examples in train set
    noTr = noTr_positives + (noTr_negatives) #total size of train set

    #only sample from first 90% of dataset; shuffle first
    data1=[]
    f=open('All_hyperparam_training/icsd_full_data_unique_no_frac_1990_nopenta.txt')
    i=0
    for line in f:
        if(i>noTr_positives and i<noTr_positives*1.05):
            data1.append(line.replace('\n',''))
        i+=1
    f.close()

    data0=[]
    f=open('./Datasets/full_unsynthesized_examples.txt')
    i=0
    for line in f:
        if(i>noTr_negatives and i<noTr_negatives*1.05):
            data0.append(line.replace('\n',''))
        i+=1
    f.close()

    #shuffle the positive and negative examples with themselves
    negative_indices=list(range(0,len(data0)))
    random.shuffle(negative_indices)
    positive_indices=list(range(0,len(data1)))
    random.shuffle(positive_indices)
    data0=np.array(data0)
    data1=np.array(data1)
    data0=data0[negative_indices]
    data1=data1[positive_indices]
    featmat0=get_features(data0)
    featmat1=get_features(data1)

    #get labels
    labs = np.zeros((len(data0) + len(data1),1))
    for ind,ent in enumerate(data1):
        labs[ind,0] = 1
    unique, counts = np.unique(labs, return_counts=True)
    ind0 = np.where(labs==0)[0] #indices of label=0
    ind1 = np.where(labs==1)[0] #indices of label=1
    #print(len(ind0),len(ind1))

    #combine positives and negatives and shuffle
    featmat3 = np.concatenate((featmat0,featmat1)) #set legths of labels 0 and 1 to be the same in the new feature matrix featmat3
    datasorted = np.concatenate((data0,data1)) #data ordered the same as featmat3
    labs3 = np.concatenate((labs[ind0], labs[ind1]), axis=0) #labels ordered the same as featmat3
    noS = len(featmat3)
    ind = list(range(0,noS)) #training set index
    random.shuffle(ind) #shuffle training set index
    #indB = list(range(0,noTr_subset)) #used later for batch
    labs3 = np.column_stack((labs3,np.abs(labs3-1)))
    xtr_batch = featmat3[ind[0:],:]
    ytr_batch = labs3[ind[0:],:]
    data_batch=datasorted[ind[0:]]
    return(xtr_batch, ytr_batch, data_batch)

def get_batch_val_1980(neg_positive_ratio):

    random.seed(3)
    np.random.seed(3)
    noTr_positives=15100 #number positive examples in train set
    noTr_negatives=noTr_positives*neg_positive_ratio #no. negatives examples in train set
    noTr = noTr_positives + (noTr_negatives) #total size of train set

    #only sample from first 90% of dataset; shuffle first
    data1=[]
    f=open('All_hyperparam_training/icsd_full_data_unique_no_frac_1980_nopenta.txt')
    i=0
    for line in f:
        if(i>noTr_positives and i<noTr_positives*1.05):
            data1.append(line.replace('\n',''))
        i+=1
    f.close()
    data0=[]
    f=open('./Datasets/full_unsynthesized_examples.txt')
    i=0
    for line in f:
        if(i>noTr_negatives and i<noTr_negatives*1.05):
            data0.append(line.replace('\n',''))
        i+=1
    f.close()

    #shuffle the positive and negative examples with themselves
    negative_indices=list(range(0,len(data0)))
    random.shuffle(negative_indices)
    positive_indices=list(range(0,len(data1)))
    random.shuffle(positive_indices)
    data0=np.array(data0)
    data1=np.array(data1)
    data0=data0[negative_indices]
    data1=data1[positive_indices]
    featmat0=get_features(data0)
    featmat1=get_features(data1)

    #get labels
    labs = np.zeros((len(data0) + len(data1),1))
    for ind,ent in enumerate(data1):
        labs[ind,0] = 1
    unique, counts = np.unique(labs, return_counts=True)
    ind0 = np.where(labs==0)[0] #indices of label=0
    ind1 = np.where(labs==1)[0] #indices of label=1
    #print(len(ind0),len(ind1))

    #combine positives and negatives and shuffle
    featmat3 = np.concatenate((featmat0,featmat1)) #set legths of labels 0 and 1 to be the same in the new feature matrix featmat3
    datasorted = np.concatenate((data0,data1)) #data ordered the same as featmat3
    labs3 = np.concatenate((labs[ind0], labs[ind1]), axis=0) #labels ordered the same as featmat3
    noS = len(featmat3)
    ind = list(range(0,noS)) #training set index
    random.shuffle(ind) #shuffle training set index
    #indB = list(range(0,noTr_subset)) #used later for batch
    labs3 = np.column_stack((labs3,np.abs(labs3-1)))
    xtr_batch = featmat3[ind[0:],:]
    ytr_batch = labs3[ind[0:],:]
    data_batch=datasorted[ind[0:]]
    return(xtr_batch, ytr_batch, data_batch)

def get_ox_dict(dict_name):
    #returns either the 'common' or 'full' oxidation state dictionary for all elements
    species_in_use = ['Ac', 'Ag', 'Al', 'As', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Br', 'C', 'Ca', 'Cd', 'Ce', 'Cl', 'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Er', 'Eu', 'F', 'Fe', 'Ga', 'Gd', 'Ge', 'H', 'Hf', 'Hg', 'Ho', 'I', 'In', 'Ir', 'K', 'La', 'Li', 'Lu', 'Mg', 'Mn', 'Mo', 'N', 'Na', 'Nb', 'Nd', 'Ni', 'Np', 'O', 'Os', 'P', 'Pa', 'Pb', 'Pd', 'Pr', 'Pt', 'Pu', 'Rb', 'Re', 'Rh', 'Ru', 'S', 'Sb', 'Sc', 'Se', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb', 'Te', 'Th', 'Ti', 'Tl', 'Tm', 'U', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr']
    element_names_array=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl', 'Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu']

    if(dict_name=='common'):
        mg_common_ox_dict = {}
        for specie in species_in_use:
            mg_common_ox_dict[specie] = list(mg.core.periodic_table.Specie(specie).common_oxidation_states)
        common_ox_dict = mg_common_ox_dict
        common_ox_dict = {x: common_ox_dict[x] for x in common_ox_dict if x in species_in_use}
        return(common_ox_dict)
    elif(dict_name=='full'):
        mg_full_ox_dict = {}
        for specie in species_in_use:
            mg_full_ox_dict[specie] = list(mg.core.periodic_table.Specie(specie).oxidation_states)
        full_ox_dict = mg_full_ox_dict
        full_ox_dict = {x: full_ox_dict[x] for x in full_ox_dict if x in species_in_use}
        return(full_ox_dict)
