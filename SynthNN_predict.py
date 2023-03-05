import numpy as np
import matplotlib.pyplot as plt
import pymatgen as mg
import tensorflow as tf
import re
import argparse
from pymatgen.core.periodic_table import Element

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

def SynthNN_best_user_model(data_input):
    #predicts the synthesizability using the best performing SynthNN model
    num_positive=41599
    x_input=get_features(data_input)
    y_input=np.zeros((len(data_input),2))
    neg_pos_ratio=25
    weight_for_0 = (1 + neg_pos_ratio) / (2*neg_pos_ratio)
    weight_for_1 = (1 + neg_pos_ratio) / (2*1)
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
    return(preds[:,0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get SynthNN Predictions')
    parser.add_argument('--input_file', metavar='path', required=True, help='Path to Input File')
    parser.add_argument('--output_file', metavar='path', required=True, help='Path to Output File')
    args = parser.parse_args()
    data=np.loadtxt(args.input_file,dtype=str)
    preds=SynthNN_best_user_model(data)
    outfile= open(args.output_file, 'w')
    for i in range(len(data)):
        outfile.write(data[i] + ',' + str(preds[i]) + '\n')
    outfile.close()
