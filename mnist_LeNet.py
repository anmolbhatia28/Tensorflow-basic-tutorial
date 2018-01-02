from tensorflow.contrib.layers import flatten
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 1024])
y_ = tf.placeholder(tf.float32, shape=[None, 10])


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#modified conv2d for LeNet-5
def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

#modified conv2d for LeNet-5
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')    

###########################LeNet-5 architecture#################################
mu = 0
sigma = 0.1


#Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6
W_conv1 = weight_variable([5, 5, 1, 6])
b_conv1 = bias_variable([6])
x_image = tf.reshape(x, [-1, 32, 32, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

#Pooling. Input = 28x28x6. Output = 14x14x6.
h_pool1 = max_pool_2x2(h_conv1)

#Layer 2: Convolutional. Output = 10x10x16.
W_conv2 = weight_variable([5, 5, 6, 16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

#Pooling. Input = 10x10x16. Output = 5x5x16.
h_pool2 = max_pool_2x2(h_conv2)

# Flatten. Input = 5x5x16. Output = 400.
fc1 = flatten(h_pool2)
    
#Layer 3: Fully Connected. Input = 400. Output = 120.
fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = mu, stddev = sigma))
fc1_b = tf.Variable(tf.zeros(120))
fc1 = tf.matmul(fc1,fc1_w) + fc1_b
    
#Activation.
fc1 = tf.nn.relu(fc1)

#Layer 4: Fully Connected. Input = 120. Output = 84.
fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = mu, stddev = sigma))
fc2_b = tf.Variable(tf.zeros(84))
fc2 = tf.matmul(fc1,fc2_w) + fc2_b
    
#Activation.
fc2 = tf.nn.relu(fc2)
    
#Fully Connected. Input = 84. Output = 10.
fc3_w = tf.Variable(tf.truncated_normal(shape = (84,10), mean = mu , stddev = sigma))
fc3_b = tf.Variable(tf.zeros(10))
y_conv = tf.matmul(fc2, fc3_w) + fc3_b

#small thing required later
keep_prob = tf.placeholder(tf.float32)

##############################end of LeNet-5####################################

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(16)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))