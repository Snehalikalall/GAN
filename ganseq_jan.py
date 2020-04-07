# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:26:54 2019

@author: snehalika
"""
import timeit
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
from numpy import genfromtxt
#from multiprocessing import Pool
import matplotlib.pyplot as plt
from annoy import AnnoyIndex

def annoy_main(data,obs):
     indices=np.zeros((obs,5))
     f=data.shape[1]
     t = AnnoyIndex(f, 'angular')
     for i in range(obs):
             t.add_item(i, data[i,:])

     t.build(20) # 10 trees
     t.save('test.ann')

     u = AnnoyIndex(f, 'angular')
     u.load('test.ann') # super fast, will 
     for i in range(obs):
             indices[i,:]=u.get_nns_by_item(i, 5)
         

     arr1=np.ones(obs)
     indices=indices.astype(int)
     Nb=np.zeros(4)
     m=np.zeros(4)
     for i in range(0,obs):
             if arr1[i]!=0:
                     Nb = indices[i][1:5]
                     arr1[Nb]=m
                     
                     


     return arr1


data = genfromtxt('data734.csv',delimiter=",") #Give the data path
#data = genfromtxt('/home/snehalika/Desktop/data200.csv',delimiter=",")
datan=data
dumap=np.transpose(data)
Xnew=dumap

for i in range(0,20):
	row=Xnew.shape[0]
	result=annoy_main(Xnew,row)
	c=np.nonzero(result)
	c1=c[0]
	Xnew=Xnew[c1,:]
 
Xnew=np.transpose(Xnew)
row=datan.shape[0] 
col=datan.shape[1]
result=np.zeros((datan.shape[0],col))
d1=np.round((2/3)*col)
d2=np.round((2/3)*d1)
start = timeit.default_timer()
result=np.zeros((datan.shape[0]*37,col))
k=0



for j in range(13):
        def sample_Z(m, n):
            return np.random.uniform(-1., 1., size=[m, n])

        tf.reset_default_graph()
        def generator(Z,hsize=[d1, d2],reuse=False):
            with tf.variable_scope("GAN/Generator",reuse=reuse):
                h1 = tf.layers.dense(Z,hsize[0],activation=tf.nn.leaky_relu)
                h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
                out = tf.layers.dense(h2,col)

            return out

        def discriminator(X,hsize=[d1, d2],reuse=False):
            with tf.variable_scope("GAN/Discriminator",reuse=reuse):
                h1 = tf.layers.dense(X,hsize[0],activation=tf.nn.leaky_relu)
                h2 = tf.layers.dense(h1,hsize[1],activation=tf.nn.leaky_relu)
                h3 = tf.layers.dense(h2,col)
                out = tf.layers.dense(h3,1)

            return out, h3


        X = tf.placeholder(tf.float32,[None,col])
        Z = tf.placeholder(tf.float32,[None,col])

        G_sample = generator(Z)
        r_logits, r_rep = discriminator(X)
        f_logits, g_rep = discriminator(G_sample,reuse=True)

        disc_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=r_logits,labels=tf.ones_like(r_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.zeros_like(f_logits)))
        gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=f_logits,labels=tf.ones_like(f_logits)))

        gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
        disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

        gen_step = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(gen_loss,var_list = gen_vars) # G Train step
        disc_step = tf.train.AdamOptimizer(learning_rate=0.0003).minimize(disc_loss,var_list = disc_vars) # D Train step


        #x_plot=TSNE(n_components=2).fit_transform(datan)

        sess = tf.Session()
        tf.global_variables_initializer().run(session=sess)
        f = open('loss_logs734.csv','w')
        f.write('Iteration,Discriminator Loss,Generator Loss\n')

        bs= datan.shape[0]
        nd_steps = 10
        ng_steps = 10
        for i in range(2501):
            X_batch = datan
            col1=col-Xnew.shape[1]
            da1= sample_Z(bs,col1)
            Z_batch=np.column_stack((da1,Xnew))
            Z_batch = sample_Z(bs,col)
            
            for _ in range(nd_steps):
                 _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})
            		
            for _ in range(ng_steps):
                _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})
            
            
            if i%10 == 0:
                f.write("%d,%f,%f\n"%(i,dloss,gloss))
    
            rrep_gstep, grep_gstep = sess.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})
            #print "Iterations: %d\t Discriminator loss: %.4f\t Generator loss: %.4f"%(i,dloss,gloss)
            g_plot = sess.run(G_sample, feed_dict={Z: Z_batch})
            #count=g_plot.shape[0]
#            if i%1000 == 0:
#                gtsne=TSNE(n_components=2).fit_transform(g_plot)
#                plt.figure()
#                xax = plt.scatter(x_plot[:,0], x_plot[:,1])
#                gax = plt.scatter(gtsne[:,0],gtsne[:,1])
#                plt.legend((xax,gax), ("Real Data","Generated Data"))
#                plt.title('Samples at Iteration %d'%i)
#                plt.tight_layout()
#                plt.savefig('iteration_%d.png'%i)
#                plt.close()
        print(j)    
        result[k:(k+row),:]=g_plot
        k=k+row
 
    

stop = timeit.default_timer()
print('Time: ', stop - start) 
np.savetxt("result734_cond.csv",result, delimiter=",")  #Give the save file path
print("Size of generated Data1",result.shape)






