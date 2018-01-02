from tensorflow.contrib.layers import flatten
from tensorflow.examples.tutorials.mnist import input_data

#mnist is a standard dataset, so already included in tensor-flow, other datasets such as cifar, etc. can be loaded in a similar way
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tenFlow

# just initializing variables
# x has 784 because mnist images are of size 28*28. now, depending on batch size, the variable x will be  " batch_size *  784 "
x = tenFlow.placeholder(tenFlow.float32, shape=[None, 784])
# similarly y will be of the form " batch_size * 10 ", 10: because, there are 10 classes
y_ = tenFlow.placeholder(tenFlow.float32, shape=[None, 10])

#these functions will be used in our neural net
def get_weight_holder(shape):
  #below given line adds a small amount of noise to the weighrs, if noise not added, then gradient will always be 0
  initial = tenFlow.truncated_normal(shape, stddev=0.1)
  return tenFlow.Variable(initial)

def get_bias_holder(shape):
  #similarly small amount of noise is added to bias
  initial = tenFlow.constant(0.1, shape=shape)
  return tenFlow.Variable(initial)

def conv2d(x, W):
  return tenFlow.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
  

def max_pool_2x2(x):
  return tenFlow.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


##########################2 convolution layer network###########################
#source of formulas : http://cs231n.github.io/convolutional-networks/

#Layer 1: Convolutional. Input = 28x28x1(MNIST Dataset) Output = 28x28x32. Here 1 implies input channels, since it is b&w image, channel is 1, if it was any colored image, it would be 3(R,G,B)
W_conv1 = get_weight_holder([5, 5, 1, 32])
# W'=(W−F+2P)/S+1, where W is the input size, F is the convolving window size, P is the padding(here 'SAME' padding implies padding = 2) and S is the stride(movement, which is 1,1,1,1 i.e. window will move pixel by pixel, first in x-direction, then in y-direction) so, (28- 5 + 2*2)/1 +1 = 28 
b_conv1 = get_bias_holder([32])
x_image = tenFlow.reshape(x, [-1, 28, 28, 1])
#image is reshaped to 4-d vector so that it can be processed by convolutional layer(just a requirement for TensorFlow, no other specific reason)
h_conv1 = tenFlow.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

#max - pooling is applied, now the images are resized to 14*14, total output is 14*14*32
h_pool1 = max_pool_2x2(h_conv1)

#Convolutional Layer 2 : Input = 14*14*32, Output: 14*14*64
W_conv2 = get_weight_holder([5, 5, 32, 64])
b_conv2 = get_bias_holder([64])
h_conv2 = tenFlow.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

#Pooling Layer 2: Input = 14*14*64, Output: 7*7*64 
h_pool2 = max_pool_2x2(h_conv2)

#the below given variables are weight holders for relu - layer
W_fc1 = get_weight_holder([7 * 7 * 64, 1024])
b_fc1 = get_bias_holder([1024])

#the below line flattens the image so that the data is 1-d, no of rows = 1, no of columns = 7*7*64
h_pool2_flat = tenFlow.reshape(h_pool2, [-1, 7*7*64])

#relu layer
h_fc1 = tenFlow.nn.relu(tenFlow.matmul(h_pool2_flat, W_fc1) + b_fc1)

#dropout is used for over-fitting,  placeholder(keep_prob) for the probability that a neuron's output is kept during dropout. This allows us to turn dropout on during training, and turn it off during testing.
keep_prob = tenFlow.placeholder(tenFlow.float32)
h_fc1_drop = tenFlow.nn.dropout(h_fc1, keep_prob)

#below are the weights for last layer
W_fc2 = get_weight_holder([1024, 10])
b_fc2 = get_bias_holder([10])

#the final layer, containing probabilies with respect to each class
y_conv = tenFlow.matmul(h_fc1_drop, W_fc2) + b_fc2

###########################End of 2 Convolutional network#######################

cross_entropy = tenFlow.reduce_mean(tenFlow.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
#defining the loss, the loss we have used is cross-entropy loss. Majorly in all problems, cross-entropy loss is used
# https://en.wikipedia.org/wiki/Cross_entropy

train_step = tenFlow.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#Adam-optimizer is an extension to SGD, SGD could also be used here, the loss which is to be minimized is put in brackets

correct_prediction = tenFlow.equal(tenFlow.argmax(y_conv, 1), tenFlow.argmax(y_, 1))
accuracy = tenFlow.reduce_mean(tenFlow.cast(correct_prediction, tenFlow.float32))

#write accuracy to file
file = open('./output/results.txt','w+')
accuracies = []
with tenFlow.Session() as sess:
    sess.run(tenFlow.global_variables_initializer())
    for i in range(5000):
        batch = mnist.train.next_batch(50)
        print('len of batch is ' + str(len(batch)))
        print(len(batch[0]))
        print(len(batch[1]))
        print(type(batch[0][0]))
        print(batch[1][0])

        #above line gets batch of images, each batch split into 50 images, batch[0] stores images and batch[1] stores labels
        
        #get accuracy for each iteration and store
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        accuracies.append(train_accuracy)
        
        if i % 100 == 0:
            #print accuracy only if i is a multiple of 100, just to keep it clean
            print('step %d, training accuracy %g' % (i, train_accuracy))
        # below line runs for each batch 20000 times
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    
    #print final accuracy
    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

#store the accuracies for each iteration in a file
for item in accuracies:
  file.write(str(item) + '\n')