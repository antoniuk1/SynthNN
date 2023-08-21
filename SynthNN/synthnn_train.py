import numpy as np
import matplotlib.pyplot as plt
from io import StringIO 
import re
import pymatgen as mg
import tensorflow as tf
import random
import os
from pymatgen.core.periodic_table import Element
import linecache
from SynthNN.utils import load_data,get_batch_val,get_features,random_lines,SemiTransform,get_batch,perf_measure
import uuid
import json

def train_synthNN(params_dict):
    #hyperparameters
    np.random.seed()
    random.seed()
    M = params_dict['M']
    tf.compat.v1.disable_eager_execution()
    tstep=params_dict['tstep']
    num_steps=params_dict['num_steps']
    no_h1 = params_dict['no_h1']
    no_h2= params_dict['no_h2']
    batch_size = params_dict['batch_size']
    semi_starting=params_dict['semi_starting']
    output_dir=params_dict['output_dir_name']
    train_split=params_dict['train_split']
    val_split=params_dict['val_split']
    test_split=params_dict['test_split']

    job_id_code=str(uuid.uuid4())
    #make directory for storing model weights
    isExistingOutDir = os.path.exists(output_dir)
    if not isExistingOutDir:
        os.makedirs(output_dir)
    negative_weight_output_file=output_dir+"semi_weights_testing_neg_30M" + job_id_code +  ".txt"

   
    #Save hyperparameter dictionary to file
    json.dump(params_dict,open(output_dir + 'hyperparams' + job_id_code + '.json','w'))
    with open(params_dict['negative_example_file_path'], 'r') as fp:
        num_neg_lines = sum(1 for line in fp)
        print('Found ' +str(num_neg_lines)+ ' negative examples.')
    fp.close()
    with open(params_dict['positive_example_file_path'], 'r') as fp:
        num_pos_lines = sum(1 for line in fp)
        print('Found ' +str(num_pos_lines)+ ' positive examples.')

    fp.close()
    neg_pos_ratio=int(np.floor(num_neg_lines/num_pos_lines))
    print('Using a Nsynth Ratio of: ' + str(neg_pos_ratio))
    weight_for_0 = (1 + neg_pos_ratio) / (2*neg_pos_ratio)
    weight_for_1 = (1 + neg_pos_ratio) / (2*1)


    xtr,ytr,batch_data,weights,idxs=get_batch(batch_size, neg_pos_ratio, use_semi_weights=False, num_pos_lines=num_pos_lines,train_split=train_split,model_name=job_id_code, pos_file_name=params_dict['positive_example_file_path'],neg_file_name=params_dict['negative_example_file_path'],output_dir=output_dir,seed=False, seed_value=0)
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, xtr.shape[1]])
    y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])

    W1 = tf.Variable(tf.random.truncated_normal([xtr.shape[1],M],0,3)) #shape, mean, std
    F1 = tf.Variable(tf.random.truncated_normal([M,no_h1],0,1))
    F2 = tf.Variable(tf.random.truncated_normal([no_h1,no_h2],0,1))
    F3 = tf.Variable(tf.random.truncated_normal([no_h2,2],0,1))
    b1 = tf.Variable(tf.random.truncated_normal([no_h1],0,1))
    b2 = tf.Variable(tf.random.truncated_normal([no_h2],0,1))
    b3 = tf.Variable(tf.random.truncated_normal([2],0,1))

    semi_weights=tf.compat.v1.placeholder(tf.float32, shape=[None,1])

    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.initialize_all_variables())

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
    cross_entropy = -tf.reduce_sum(tf.multiply(y_*tf.math.log(clipped_y)*np.array([weight_for_1,weight_for_0]),semi_weights))
    correct_prediction = tf.equal(tf.argmax(a3,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    performance_array=[]
    loss_array=[]
    element_sum=np.zeros((94,1))
    epoch_counter=0
    xtr_new=xtr
    ytr_new=ytr

    train_step = tf.compat.v1.train.AdamOptimizer(tstep).minimize(cross_entropy)
    sess.run(tf.compat.v1.initialize_all_variables())
    full_weights=np.ones((len(ytr),1))
    best_perf=0
    xval,yval,val_data=get_batch_val(neg_pos_ratio,train_split,val_split,num_pos_lines,pos_file_name=params_dict['positive_example_file_path'],neg_file_name=params_dict['negative_example_file_path'])
    W1_val=[]
    F1_val=[]
    F2_val=[]
    F3_val=[]
    b1_val=[]
    b2_val=[]
    b3_val=[]
    print('Model Initial Training')

    for i in range(semi_starting):  
        epoch_counter=epoch_counter+1           
        batchx,batchy,batch_data,weights,idxs=get_batch(batch_size, neg_pos_ratio, use_semi_weights=False, num_pos_lines=num_pos_lines,train_split=train_split,model_name=job_id_code, pos_file_name=params_dict['positive_example_file_path'],neg_file_name=params_dict['negative_example_file_path'],output_dir=output_dir,seed=False, seed_value=0)
        indB = list(range(0,len(xtr_new)))
        random.shuffle(indB)
        current_weights=full_weights[indB[0:batch_size],:] 
        train_step.run(feed_dict={x: batchx, y_: batchy, semi_weights: current_weights})
        if(i%1000==0):  
            preds=a3.eval(feed_dict={x: xval, y_: yval, semi_weights: full_weights})
            TP, FP, TN, FN=perf_measure(np.array(yval)[:,0],np.array(preds)[:,0])
            val_accuracy=accuracy.eval(feed_dict={x: xval, y_: yval, semi_weights: current_weights})
            train_accuracy=accuracy.eval(feed_dict={x: batchx, y_: batchy, semi_weights: current_weights})
            performance_array.append([train_accuracy,val_accuracy, TP, FP, TN, FN])
            print('On Step #' + str(i) + '/' + str(semi_starting))
            print('Train Acc., Validation Acc., #TP, #FP, #TN, #FN')
            print([train_accuracy,val_accuracy, TP, FP, TN, FN])
            np.savetxt(output_dir + 'performance_matrix_TL_v3_pretrain' + job_id_code + '.txt',performance_array, fmt='%s')
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
    file_output = open(output_dir+"semi_weights_testing_neg_30M" + job_id_code +  ".txt","a")
    file_negatives=open(params_dict['negative_example_file_path'],'r')
    Lines = file_negatives.readlines()

    for i in range(int(np.ceil(num_neg_lines/10000))):
        with open(params_dict['negative_example_file_path']) as input_file:
            batch=input_file.readlines()[10000*(i):10000*(i+1)]
            for j in range(len(batch)):
                batch[j]=batch[j].replace('\n','')        
        xtr=get_features(batch)
        ytr=[[0,1] for j in range(len(batch))]
        pred=a3.eval(feed_dict={x: xtr, y_: ytr, semi_weights: current_weights, W1:W1_val, F1:F1_val, F2:F2_val, F3:F3_val, b1:b1_val, b2:b2_val, b3:b3_val })
        with open(negative_weight_output_file,"a") as file_output:
            for j in range(len(batch)):
                file_output.write(batch[j] +  ' ' + str(pred[j][0])+ '\n')    
    sess.close()

    print('Doing Semi-supervised Learning')
    np.random.seed()
    random.seed()
    M = 30
    x = tf.compat.v1.placeholder(tf.float32, shape=[None, xtr.shape[1]])
    y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])

    W1 = tf.Variable(tf.random.truncated_normal([xtr.shape[1],M],0,3)) #shape, mean, std
    F1 = tf.Variable(tf.random.truncated_normal([M,no_h1],0,1))
    F2 = tf.Variable(tf.random.truncated_normal([no_h1,no_h2],0,1))
    F3 = tf.Variable(tf.random.truncated_normal([no_h2,2],0,1))
    b1 = tf.Variable(tf.random.truncated_normal([no_h1],0,1))
    b2 = tf.Variable(tf.random.truncated_normal([no_h2],0,1))
    b3 = tf.Variable(tf.random.truncated_normal([2],0,1))
    semi_weights=tf.compat.v1.placeholder(tf.float32, shape=[None,1])

    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.initialize_all_variables())

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
    cross_entropy = -tf.reduce_sum(tf.multiply(y_*tf.math.log(clipped_y),semi_weights))
    correct_prediction = tf.equal(tf.argmax(a3,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    performance_array=[]
    loss_array=[]
    element_sum=np.zeros((94,1))
    epoch_counter=0
    best_perf=0
    train_step = tf.compat.v1.train.AdamOptimizer(tstep).minimize(cross_entropy)
    sess.run(tf.compat.v1.initialize_all_variables())

    for i in range(num_steps):   
        epoch_counter=epoch_counter+1           
        batchx,batchy,batch_data,weights,idxs=get_batch(batch_size, neg_pos_ratio, use_semi_weights=True, num_pos_lines=num_pos_lines,train_split=train_split,model_name=job_id_code, pos_file_name=params_dict['positive_example_file_path'],neg_file_name=params_dict['negative_example_file_path'],output_dir=output_dir,seed=False, seed_value=0)
        weights=np.reshape(weights, [len(weights),1])        
        train_step.run(feed_dict={x: batchx, y_: batchy, semi_weights: weights})
        #loss_array.append([])
        if(i%1000==0):
            preds=a3.eval(feed_dict={x: xval, y_: yval, semi_weights: weights})
            TP, FP, TN, FN=perf_measure(np.array(yval)[:,0],np.array(preds)[:,0])
            val_accuracy=accuracy.eval(feed_dict={x: xval, y_: yval, semi_weights: weights})
            train_accuracy=accuracy.eval(feed_dict={x: batchx, y_: batchy, semi_weights: weights})
            performance_array.append([train_accuracy,val_accuracy, TP, FP, TN, FN])
            print([train_accuracy,val_accuracy, TP, FP, TN, FN])
            np.savetxt(output_dir+'performance_matrix_TL_v3_30M_' + job_id_code+ '.txt',performance_array, fmt='%s')

        if(i%1000==0):
            if(val_accuracy>best_perf):
                model_name='30M_synth_v3_semi' + job_id_code + '.txt'
                best_perf=val_accuracy
                W1_val=sess.run(W1)
                F1_val=sess.run(F1)
                F2_val=sess.run(F2)
                F3_val=sess.run(F3)
                b1_val=sess.run(b1)
                b2_val=sess.run(b2)
                b3_val=sess.run(b3)
                np.savetxt(output_dir+'W1_' + model_name, W1_val)
                np.savetxt(output_dir+'F1_' + model_name, F1_val)
                np.savetxt(output_dir+'F2_' + model_name, F2_val)
                np.savetxt(output_dir+'F3_' + model_name, F3_val)
                np.savetxt(output_dir+'b1_' + model_name, b1_val)
                np.savetxt(output_dir+'b2_' + model_name, b2_val)
                np.savetxt(output_dir+'b3_' + model_name, b3_val)
    sess.close()
    return()
