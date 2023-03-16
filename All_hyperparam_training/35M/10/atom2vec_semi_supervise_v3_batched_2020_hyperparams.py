# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from io import StringIO 
import re
import pymatgen as mg
import tensorflow as tf
import random
import os
from pymatgen.core.periodic_table import Element


# In[2]:


def weighted_F1(TP,FP,TN,FN, num_negative, num_positive=41599):
    positive_precision=TP/(TP+FP)
    positive_recall=TP/(TP+FN)
    positive_F1=(2*positive_precision*positive_recall)/(positive_precision+positive_recall)
    
    negative_precision=TN/(TN+FN)
    negative_recall=TN/(TN+FP)
    negative_F1=(2*negative_precision*negative_recall)/(negative_precision+negative_recall)
    
    weighted_F1=(((num_positive)/(num_positive+num_negative))*positive_F1) + (((num_negative)/(num_positive+num_negative))*negative_F1)
    return(weighted_F1)


# In[3]:


element_names_array=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl', 'Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu']


# In[4]:


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


# In[5]:


def make_negative_data(num_examples, max_atoms=5, max_coefficient=11, seed=3, weighted=False):
    #play around with max_coefficient values!!
    
    output_array=[]
    element_names_array=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl', 'Ar','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn','Fr','Ra','Ac','Th','Pa','U','Np','Pu']
    element_sum=np.loadtxt('GNN_icsd_mpids_unique_element_proportions.txt')
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
    
    


# In[6]:


def get_features(data0):
    p = re.compile('[A-Z][a-z]?\d*\.?\d*')
    p3 = re.compile('[A-Z][a-z]?')
    p5 = re.compile('\d+\.?\d+|\d+')
    data0_ratio=[]
    for i in data0:
        
        x = mg.core.composition.Composition(i).alphabetical_formula.replace(' ','')
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


# In[7]:


def random_lines(filename, file_size, num_samples):
    idxs = random.sample(range(1,file_size), num_samples)
    return ([linecache.getline(filename, i) for i in idxs], idxs)


# In[8]:


def get_batch_val(neg_positive_ratio):

    random.seed(3)
    np.random.seed(3)

    noTr_positives=48200 #number positive examples in train set
    noTr_negatives=noTr_positives*neg_positive_ratio #no. negatives examples in train set
    noTr = noTr_positives + (noTr_negatives) #total size of train set



    #only sample from first 90% of dataset ( need to shuffle first because GNN_icsd is alphabetical!)
    data1=[]
    f=open('icsd_full_data_unique_no_frac_no_penta_2020.txt')
    i=0
    for line in f:
        if(i>noTr_positives and i<noTr_positives*1.05):
            data1.append(line.replace('\n',''))
        i+=1
    f.close()

    data0=[]
    f=open('standard_neg_ex_tr_val_v5_balanced_shuffled.txt')
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


# In[9]:


import random
import linecache

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



    #only sample from first 90% of dataset ( need to shuffle first because GNN_icsd is alphabetical!)
    data1=[]
    pulled_lines1,idxs1=random_lines('icsd_full_data_unique_no_frac_no_penta_2020.txt', noTr_positives,num_positive_examples)
    
    for line in pulled_lines1:
        data1.append(line.replace('\n',''))

    data0=[]
    pulled_lines0,idxs0=random_lines('standard_neg_ex_tr_val_v5_balanced_shuffled.txt', noTr_negatives, num_negative_examples)
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
        file=open('semi_weights_testing_pos_35M' + model_name + '.txt','r')
        content=file.readlines()
        weights1=[]
        for i in idxs1:
            weights1.append(float(content[i-1].split()[1]))
        file.close()

        weights0=[]
        file=open('semi_weights_testing_neg_35M' + model_name + '.txt','r')
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



# In[10]:


#weight_for_0=1
#weight_for_1=1
#THESE SEEM BACKWARDS?, increasing weight_for_1 gives way more false positives, less true negatives, less FN

neg_pos_ratio=10
weight_for_0 = (1 + neg_pos_ratio) / (2*neg_pos_ratio)
weight_for_1 = (1 + neg_pos_ratio) / (2*1)
print(weight_for_0, weight_for_1)


# In[11]:


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


# In[12]:


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


# In[13]:


#hyperparameters
np.random.seed()
random.seed()
M = 35

randint=np.random.randint(0,5,5)
tstep=[1e-2, 5e-3, 2e-3, 5e-4, 2e-4][randint[0]]
num_steps=800000
no_h1 = [30,40,50,60,80][randint[1]]
no_h2= [30,40,50,60,80][randint[2]]
batch_size = [512,512,1024,1024,1024][randint[3]]
semi_starting=[20000,40000,60000,80000,100000][randint[4]]

model_name_params=str(randint[0]) + str(randint[1]) + str(randint[2]) + str(randint[3]) + str(randint[4])

xtr,ytr,batch_data,weights,idxs=get_batch(batch_size, neg_pos_ratio, use_semi_weights=False, model_name=model_name_params)

x = tf.placeholder(tf.float32, shape=[None, xtr.shape[1]])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

#USE THIS IF NEW JOB
W1 = tf.Variable(tf.truncated_normal([xtr.shape[1],M],0,3)) #shape, mean, std
#W1 = tf.placeholder(tf.float32, shape=[xtr.shape[1],M]) #shape, mean, std

F1 = tf.Variable(tf.truncated_normal([M,no_h1],0,1))
F2 = tf.Variable(tf.truncated_normal([no_h1,no_h2],0,1))
F3 = tf.Variable(tf.truncated_normal([no_h2,2],0,1))

b1 = tf.Variable(tf.truncated_normal([no_h1],0,1))
b2 = tf.Variable(tf.truncated_normal([no_h2],0,1))
b3 = tf.Variable(tf.truncated_normal([2],0,1))

semi_weights=tf.placeholder(tf.float32, shape=[None,1])



#W1_loaded=np.loadtxt('TL_10M_ceder_form_energy_hyperparam/W1_10M_formation_energies_v3_2234.txt') #load weights if doing TL HERE!!


sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

z0_raw = tf.multiply(tf.expand_dims(x,2),tf.expand_dims(W1,0)) #(ntr, I, M)
#yraw = tf.mul(tf.expand_dims(x,2),tf.expand_dims(W1,0))
tempmean,var = tf.nn.moments(z0_raw,axes=[1])
# y = tf.concat(1,[tf.reduce_sum(y,1),var,tf.reduce_max(y,1),tf.reduce_min(y,1)])
z0 = tf.concat([tf.reduce_sum(z0_raw,1)],1) #(ntr, M)

z1 = tf.add(tf.matmul(z0,F1),b1) #(ntr, no_h1)
a1 = tf.tanh(z1) #(ntr, no_h1)

z2= tf.add(tf.matmul(a1,F2),b2) #(ntr,no_h1)
a2= tf.tanh(z2) #(ntr, no_h1)

z3 = tf.add(tf.matmul(a2,F3),b3) #(ntr, 2)
a3 = tf.nn.softmax(z3) #(ntr, 2)

clipped_y = tf.clip_by_value(a3, 1e-10, 1.0)

cross_entropy = -tf.reduce_sum(tf.multiply(y_*tf.log(clipped_y)*np.array([weight_for_1,weight_for_0]),semi_weights))

#try adding regularization
#reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#loss = tf.add(cross_entropy , tf.multiply(reg_constant , tf.reduce_sum(reg_losses)))

#cross_entropy=(-1*(y_[:,0]*tf.log(clipped_y[:,0]))) + ((1-y_[:,0])*(tf.log(1-clipped_y[:,0])))


correct_prediction = tf.equal(tf.argmax(a3,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
performance_array=[]
loss_array=[]
element_sum=np.zeros((94,1))

#decay=0.999
epoch_counter=0
xtr_new=xtr
ytr_new=ytr

train_step = tf.train.AdamOptimizer(tstep).minimize(cross_entropy)
sess.run(tf.initialize_all_variables())
full_weights=np.ones((len(ytr),1))
best_perf=0
xval,yval,val_data=get_batch_val(neg_pos_ratio)

W1_val=[]
F1_val=[]
F2_val=[]
F3_val=[]
b1_val=[]
b2_val=[]
b3_val=[]

for i in range(semi_starting): 
    
    epoch_counter=epoch_counter+1           

    batchx,batchy,batch_data,weights,idxs=get_batch(batch_size, neg_pos_ratio, use_semi_weights=False, model_name=model_name_params)
    
    #batchx = xtr_new[indB[0:batch_size],:]
    #batchy = ytr_new[indB[0:batch_size],:]
    
    indB = list(range(0,len(xtr_new)))
    random.shuffle(indB)
    current_weights=full_weights[indB[0:batch_size],:]
    
    train_step.run(feed_dict={x: batchx, y_: batchy, semi_weights: current_weights})
    #loss_array.append([])
    if(i%1000==0):
        
        preds=a3.eval(feed_dict={x: xval, y_: yval, semi_weights: full_weights})
        TP, FP, TN, FN=perf_measure(np.array(yval)[:,0],np.array(preds)[:,0])
        val_accuracy=accuracy.eval(feed_dict={x: xval, y_: yval, semi_weights: current_weights})
        train_accuracy=accuracy.eval(feed_dict={x: batchx, y_: batchy, semi_weights: current_weights})
        performance_array.append([train_accuracy,val_accuracy, TP, FP, TN, FN])
        #np.savetxt('performance_matrix_TL_v3_' + str(randint[0]) + str(randint[1]) + str(randint[2]) + str(randint[3]) + '.txt',performance_array, fmt='%s')
        if(val_accuracy>best_perf):
            best_perf=val_accuracy
            W1_val=sess.run(W1)
            F1_val=sess.run(F1)
            F2_val=sess.run(F2)
            F3_val=sess.run(F3)
            b1_val=sess.run(b1)
            b2_val=sess.run(b2)
            b3_val=sess.run(b3)

        
#print out all preds to a file (for weighting for semi-supervised learning)
file_output = open("semi_weights_testing_pos_35M" + model_name_params +  ".txt","a")
file_positives=open('icsd_full_data_unique_no_frac_no_penta_2020.txt', 'r')
Lines = file_positives.readlines()
for line in Lines:
    xtr=get_features([line.replace('\n','')])
    ytr=[[0,1]]
    pred=a3.eval(feed_dict={x: xtr, y_: ytr, semi_weights: current_weights, W1:W1_val, F1:F1_val, F2:F2_val, F3:F3_val, b1:b1_val, b2:b2_val, b3:b3_val })
    file_output.write(line.replace('\n','') +  ' ' + str(pred[0][0]) + '\n')   
file_positives.close()
file_output.close()

file_output = open("semi_weights_testing_neg_35M" + model_name_params +  ".txt","a")
file_negatives=open('standard_neg_ex_tr_val_v5_balanced_shuffled.txt','r')
Lines = file_negatives.readlines()
for line in Lines:
    xtr=get_features([line.replace('\n','')])
    ytr=[[0,1]]
    pred=a3.eval(feed_dict={x: xtr, y_: ytr, semi_weights: current_weights, W1:W1_val, F1:F1_val, F2:F2_val, F3:F3_val, b1:b1_val, b2:b2_val, b3:b3_val })
    file_output.write(line.replace('\n','') +  ' ' + str(pred[0][0])+ '\n') 

file_negatives.close()
file_output.close()
sess.close()


np.random.seed()
random.seed()
M = 35

x = tf.placeholder(tf.float32, shape=[None, xtr.shape[1]])
y_ = tf.placeholder(tf.float32, shape=[None, 2])

#USE THIS IF NEW JOB
W1 = tf.Variable(tf.truncated_normal([xtr.shape[1],M],0,3)) #shape, mean, std
#W1 = tf.placeholder(tf.float32, shape=[xtr.shape[1],M]) #shape, mean, std

F1 = tf.Variable(tf.truncated_normal([M,no_h1],0,1))
F2 = tf.Variable(tf.truncated_normal([no_h1,no_h2],0,1))
F3 = tf.Variable(tf.truncated_normal([no_h2,2],0,1))

b1 = tf.Variable(tf.truncated_normal([no_h1],0,1))
b2 = tf.Variable(tf.truncated_normal([no_h2],0,1))
b3 = tf.Variable(tf.truncated_normal([2],0,1))

semi_weights=tf.placeholder(tf.float32, shape=[None,1])
#W1_loaded=np.loadtxt('TL_10M_ceder_form_energy_hyperparam/W1_10M_formation_energies_v3_2234.txt') #load weights if doing TL HERE!!

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

z0_raw = tf.multiply(tf.expand_dims(x,2),tf.expand_dims(W1,0)) #(ntr, I, M)
tempmean,var = tf.nn.moments(z0_raw,axes=[1])
z0 = tf.concat([tf.reduce_sum(z0_raw,1)],1) #(ntr, M)
z1 = tf.add(tf.matmul(z0,F1),b1) #(ntr, no_h1)
a1 = tf.tanh(z1) #(ntr, no_h1)
z2= tf.add(tf.matmul(a1,F2),b2) #(ntr,no_h1)
a2= tf.tanh(z2) #(ntr, no_h1)
z3 = tf.add(tf.matmul(a2,F3),b3) #(ntr, 2)
a3 = tf.nn.softmax(z3) #(ntr, 2)

clipped_y = tf.clip_by_value(a3, 1e-10, 1.0)
cross_entropy = -tf.reduce_sum(tf.multiply(y_*tf.log(clipped_y),semi_weights))

#try adding regularization
#reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#loss = tf.add(cross_entropy , tf.multiply(reg_constant , tf.reduce_sum(reg_losses)))
#cross_entropy=(-1*(y_[:,0]*tf.log(clipped_y[:,0]))) + ((1-y_[:,0])*(tf.log(1-clipped_y[:,0])))

correct_prediction = tf.equal(tf.argmax(a3,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
performance_array=[]
loss_array=[]
element_sum=np.zeros((94,1))
epoch_counter=0
best_perf=0

train_step = tf.train.AdamOptimizer(tstep).minimize(cross_entropy)
sess.run(tf.initialize_all_variables())

for i in range(num_steps):   
    epoch_counter=epoch_counter+1           
    batchx,batchy,batch_data,weights,idxs=get_batch(batch_size, neg_pos_ratio, use_semi_weights=True, model_name=model_name_params)
    weights=np.reshape(weights, [len(weights),1])        
    train_step.run(feed_dict={x: batchx, y_: batchy, semi_weights: weights})
    #loss_array.append([])
    if(i%1000==0):
        preds=a3.eval(feed_dict={x: xval, y_: yval, semi_weights: weights})
        TP, FP, TN, FN=perf_measure(np.array(yval)[:,0],np.array(preds)[:,0])
        val_accuracy=accuracy.eval(feed_dict={x: xval, y_: yval, semi_weights: weights})
        train_accuracy=accuracy.eval(feed_dict={x: batchx, y_: batchy, semi_weights: weights})
        performance_array.append([train_accuracy,val_accuracy, TP, FP, TN, FN])
        #print([train_accuracy,val_accuracy, TP, FP, TN, FN])
        np.savetxt('performance_matrix_TL_v3_35M_' + model_name_params+ '.txt',performance_array, fmt='%s')

    if(i%1000==0):
        if(val_accuracy>best_perf):
            model_name='35M_synth_v3_semi' + model_name_params + '.txt'
            best_perf=val_accuracy
            W1_val=sess.run(W1)
            F1_val=sess.run(F1)
            F2_val=sess.run(F2)
            F3_val=sess.run(F3)
            b1_val=sess.run(b1)
            b2_val=sess.run(b2)
            b3_val=sess.run(b3)
            np.savetxt('W1_' + model_name, W1_val)
            np.savetxt('F1_' + model_name, F1_val)
            np.savetxt('F2_' + model_name, F2_val)
            np.savetxt('F3_' + model_name, F3_val)
            np.savetxt('b1_' + model_name, b1_val)
            np.savetxt('b2_' + model_name, b2_val)
            np.savetxt('b3_' + model_name, b3_val)
