import pandas as pd
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import numpy as np


import os
os.chdir('G:/academics/ML2 Project/drug_discovery')

data=pd.read_csv('processed.csv')

target=data.iloc[:,-1]
target=pd.get_dummies(target)
data=pd.concat([data,target],axis=1)
train,test=train_test_split(data,test_size=0.3)
train_f=train.iloc[:,1:2049]
test_f=test.iloc[:,1:2049]
train_t=train.iloc[:,2052:2059]
test_t=test.iloc[:,2052:2059]
train_t=np.reshape(train_t,[train_t.shape[0],7])
test_t=np.reshape(test_t,[test_t.shape[0],7])



#print(data.iloc[:,-1].unique())
#7 targets
print(train_f.shape)
print(train_t.shape)

x=tf.placeholder(tf.float32,shape=[None,2048])
y_=tf.placeholder(tf.float32,shape=[None,7])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_4(x):
  return tf.nn.max_pool(x, ksize=[1, 1, 4, 1],
                        strides=[1, 1, 2, 1], padding='SAME')

W_conv1 = weight_variable([1, 8, 1, 16])
b_conv1 = bias_variable([16])

x_c = tf.reshape(x, [-1, 1, 2048, 1])

h_conv1 = tf.nn.relu(conv2d(x_c, W_conv1) + b_conv1)
h_pool1 = max_pool_4(h_conv1)

W_conv2 = weight_variable([1, 4, 16, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_4(h_conv2)


W_fc1 = weight_variable([128*64*4, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 128*64*4])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 7])
b_fc2 = bias_variable([7])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(1000):
    train_accuracy = accuracy.eval(feed_dict={x: train_f, y_: train_t, keep_prob: 1.0})
    print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: train_f, y_: train_t, keep_prob: 0.8})

  print('test accuracy %g' % accuracy.eval(feed_dict={x: test_f, y_:test_t, keep_prob: 1.0}))

#training accuracy 0.980788
#test accuracy 0.813055