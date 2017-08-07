import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

import os
os.chdir('G:/academics/ML2 Project/drug_discovery')

data=pd.read_csv('processed.csv')

target=data.iloc[:,-1].unique()
#print(data.groupby(data.iloc[:,-1]))
data0=data[data.iloc[:,-1]==target[0]]
other=data[data.iloc[:,-1]!=target[0]]


data0.index=range(data0.shape[0])
data0.iloc[:,-1]=data0.iloc[:,-1].astype('category')

other.index=range(other.shape[0])
other.iloc[:,-1]=other.iloc[:,-1].astype('category')
other_f=other.iloc[:,1:2049]

train0, test0=train_test_split(data0,test_size=0.4)
test0, valid0=train_test_split(test0,test_size=0.4)
train0_f=train0.iloc[:,1:2049]
test0_f=test0.iloc[:,1:2049]
valid0_f=valid0.iloc[:,1:2049]
train0_y=train0.iloc[:,-1]
test0_y=test0.iloc[:,-1]
valid0_y=valid0.iloc[:,-1]


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)

def sample_Z(m, n):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1., 1., size=[m, n])

# Discriminator Net
X = tf.placeholder(tf.float32, shape=[None, 2048], name='X')

D_W1 = tf.Variable(xavier_init([2048, 128]), name='D_W1')
D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')

D_W2 = tf.Variable(xavier_init([128, 1]), name='D_W2')
D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')

theta_D = [D_W1, D_W2, D_b1, D_b2]

# Generator Net
Z = tf.placeholder(tf.float32, shape=[None, 1], name='Z')

G_W1 = tf.Variable(xavier_init([1, 128]), name='G_W1')
G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')

G_W2 = tf.Variable(xavier_init([128, 2048]), name='G_W2')
G_b2 = tf.Variable(tf.zeros(shape=[2048]), name='G_b2')

theta_G = [G_W1, G_W2, G_b1, G_b2]


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))


# Only update D(X)'s parameters, so var_list = theta_D
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
# Only update G(X)'s parameters, so var_list = theta_G
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

drt=[]
drv=[]
doth=[]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for it in range(10000):
        _, D_loss_curr,D_prob = sess.run([D_solver, D_loss,D_real], feed_dict={X: train0_f, Z: sample_Z(train0_y.shape[0], 1)})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(train0_y.shape[0], 1)})
        d_r_t=tf.reduce_mean(D_real).eval(feed_dict={X: train0_f})
        d_r_v=tf.reduce_mean(D_real).eval(feed_dict={X: valid0_f})
        d_other=tf.reduce_mean(D_real).eval(feed_dict={X: other_f})
        print(it,'D_R_T',d_r_t)
        print(it,'D_R_V', d_r_v)
        print(it,'D_OTHER',d_other)
        drt.append(d_r_t)
        drv.append(d_r_v)
        doth.append(d_other)
    print('test',tf.reduce_mean(D_real).eval(feed_dict={X: test0_f}))

#D_R_V 0.829643
#D_OTHER 0.237857
#test 0.856346

plt.figure(figsize=(20,20))
plt.plot(range(10000),drt,'b',label='train real')
plt.plot(range(10000),drv,'y',label='valid real')
plt.plot(range(10000),doth,'r',label='other')
plt.ylabel('rate')
plt.xlabel('step')
plt.show()

