import numpy as np
import matplotlib.pyplot as plt
import pymatgen as mg
import tensorflow as tf
import re
import argparse
from pymatgen.core.periodic_table import Element
from io import StringIO 
import random
import os
import linecache

def get_features(data0):
    I=94
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

    featmat0 = np.zeros((len(data0_ratio),I))
    # featmat: n-hot vectors with fractions
    for idx,ent in enumerate(data0_ratio):
        for idy,at in enumerate(ent[0]):
            featmat0[idx,at-1] = ent[1][idy]/sum(ent[1])
    return(featmat0)

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

def get_batch_val(neg_positive_ratio,train_split,val_split,num_pos_lines,pos_file_name='./icsd_full_data_unique_no_frac_no_penta_2020.txt',neg_file_name='./standard_neg_ex_tr_val_v5_balanced_shuffled.txt'):
    random.seed(3)
    np.random.seed(3)
    noTr_positives=int(np.floor(num_pos_lines*train_split)) #number positive examples in train set
    noTr_negatives=noTr_positives*neg_positive_ratio #no. negatives examples in train set
    num_neg_lines=num_pos_lines*neg_positive_ratio
    noTr = noTr_positives + (noTr_negatives) #total size of train set
    data1=[]
    f=open(pos_file_name)
    i=0
    for line in f:
        if(i>(train_split*num_pos_lines) and i<((train_split+val_split)*num_pos_lines)):
            data1.append(line.replace('\n',''))
        i+=1
    f.close()

    data0=[]
    f=open(neg_file_name)
    i=0
    for line in f:
        if(i>(train_split*num_neg_lines) and i<((train_split+val_split)*num_neg_lines)):
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

def get_batch(batch_size, neg_positive_ratio, use_semi_weights, num_pos_lines,train_split,model_name, pos_file_name,neg_file_name,output_dir='./',seed=False, seed_value=0):
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
    noTr_positives=int(np.floor(num_pos_lines*train_split)) #total number of positive examples in train set
    noTr_negatives=noTr_positives*neg_positive_ratio #no. negatives examples in train set
    noTr = noTr_positives + (noTr_negatives) #total size of train set
    #only sample from first 90% of dataset 
    data1=[]
    pulled_lines1,idxs1=random_lines(pos_file_name, noTr_positives,num_positive_examples)  
    for line in pulled_lines1:
        data1.append(line.replace('\n',''))
    data0=[]
    pulled_lines0,idxs0=random_lines(neg_file_name, noTr_negatives, num_negative_examples)
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
    #print(len(ind0),len(ind1))
    #combine positives and negatives and shuffle
    featmat3 = np.concatenate((featmat0,featmat1)) #set legths of labels 0 and 1 to be the same in the new feature matrix featmat3
    datasorted = np.concatenate((data0,data1)) #data ordered the same as featmat3
    labs3 = np.concatenate((labs[ind0], labs[ind1]), axis=0) #labels ordered the same as featmat3
    idxs_full=np.concatenate((idxs0,idxs1))  
    noS = len(featmat3)
    ind = list(range(0,noS)) #training set index
    random.shuffle(ind) #shuffle training set index
    labs3 = np.column_stack((labs3,np.abs(labs3-1)))
    xtr_batch = featmat3[ind[0:],:]
    ytr_batch = labs3[ind[0:],:]
    data_batch=datasorted[ind[0:]]
    idxs_full=idxs_full[ind[0:]]  
    #all weights stuff here
    weights_full=[]
    if(use_semi_weights):
        weights1=[]
        for i in range(len(idxs1)):
            weights1.append(1)

        weights0=[]
        file=open(output_dir+'semi_weights_testing_neg_30M' + model_name + '.txt','r')
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

def random_lines(filename, file_size, num_samples):
    idxs = random.sample(range(1,file_size), num_samples)
    return ([linecache.getline(filename, i) for i in idxs], idxs)

def SemiTransform(Xtrain,Ytrain,probtrain):
    X1=Xtrain[np.where(Ytrain==1)[0]]
    X0 = Xtrain[np.where(Ytrain==0)[0]]
    Xsemi=np.row_stack((X1,X0,X0))
    prob0 = probtrain[np.where(Ytrain==0)[0]]
    Y1=Ytrain[np.where(Ytrain==1)[0]]
    Y0 = Ytrain[np.where(Ytrain==0)[0]]
    Ysemi=np.concatenate((Y1,Y0,Y0+1))
    # c=np.mean(probtrain[Ytrain == 1][:,1])
    c=np.max(probtrain[np.where(Ytrain==1)[0]])
    p=prob0
    w=p/(1-p)
    w*=(1-c)/c
    weights = np.ones(len(Ysemi))
    weights[len(Y1):len(Y1)+len(Y0)] = 1-w
    weights[len(Y1) + len(Y0):] = w
    Ysemi = np.column_stack((Ysemi,np.abs(Ysemi-1)))
    return Xsemi, Ysemi, weights

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

def synthNN_predict(data,output_file_name,saved_model_dir):
    """
    Predicts the synthesizability of array of chemical compositions using the best performing SynthNN model
    and prints results out to file.
        Parameters
        ----------
        data: Numpy array or list.
          List of input chemical compositions
        Returns
        -------
        preds: Numpy array
          Predicted synthesizability from best-performing SynthNN model.
    """
    def find_all(name, path):
        result = []
        for files in os.listdir(path):
            if(files.startswith(name)):
                result.append(os.path.join(files))
        return result
    
    def SynthNN_best_user_model(data_input):
        """
        Predicts the synthesizability of array of chemical compositions using the best performing SynthNN model.
            Parameters
            ----------
            data_input: Numpy array
              List of input chemical compositions
            Returns
            -------
            preds: Numpy array
              Predicted synthesizability from best-performing SynthNN model.
        """
        
        #Load all model weights
        W1_filename=find_all('W1', saved_model_dir)
        b1_filename=find_all('b1', saved_model_dir)
        b2_filename=find_all('b2', saved_model_dir)
        b3_filename=find_all('b3', saved_model_dir)
        F1_filename=find_all('F1',saved_model_dir)
        F2_filename=find_all('F2',saved_model_dir)
        F3_filename=find_all('F3',saved_model_dir)
        if(len(W1_filename)>1):
            print('Warning! Found Multiple Hyperparameters!')
            print('Re-run by specifying only one unique set of hyperparameters.')
            return()
        W1_loaded=np.loadtxt(saved_model_dir +W1_filename[0])
        b1_loaded=np.loadtxt(saved_model_dir +b1_filename[0])
        b2_loaded=np.loadtxt(saved_model_dir +b2_filename[0])
        b3_loaded=np.loadtxt(saved_model_dir +b3_filename[0])
        b3_loaded=np.reshape(b3_loaded, [2])
        M=np.shape(W1_loaded)[1]
        no_h1=np.shape(b1_loaded)[0]
        no_h2=np.shape(b2_loaded)[0]
        F1_loaded=np.loadtxt(saved_model_dir +F1_filename[0])
        F2_loaded=np.loadtxt(saved_model_dir +F2_filename[0])
        F3_loaded=np.loadtxt(saved_model_dir +F3_filename[0])
        F3_loaded=np.reshape(F3_loaded, [no_h2,2])
        
        #Set-up Model Architecture
        x_input=get_features(data_input)
        y_input=np.zeros((len(data_input),2))
        tf.compat.v1.disable_eager_execution()
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
        sess.run(tf.compat.v1.initialize_all_variables())
        preds=a3.eval(feed_dict={x: x_input, y_: y_input , W1:W1_loaded, F1:F1_loaded, F2:F2_loaded, F3:F3_loaded, b1:b1_loaded, b2:b2_loaded, b3:b3_loaded})     
        sess.close()
        return(preds[:,0])
    preds=SynthNN_best_user_model(data)
    if(len(preds)>0):
        outfile= open(output_file_name, 'w')
        outfile.write('formula,SynthPred(Synthesizable=1) \n')
        for i in range(len(data)):
            outfile.write(data[i] + ',' + str(preds[i]) + '\n')
        outfile.close()
        return(preds)
    else:
        return()
