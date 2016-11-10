#Some must-use packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy

from sys import argv
import tensorflow as tf

def read_my_file_format(filename_queue):
	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)
	record_defaults = [[0], ['0'*4*1024]]
	col1, col2 = tf.decode_csv(value, record_defaults=record_defaults)
	return col2, col1

def input_pipeline(filenames, batch_size, num_epochs=None):
	filename_queue = tf.train.string_input_producer(
		filenames, num_epochs=num_epochs, shuffle=True)

	example, label = read_my_file_format(filename_queue)

	# min_after_dequeue defines how big a buffer we will randomly sample
	#   from -- bigger means better shuffling but slower start up and more
	#   memory used.
	# capacity must be larger than min_after_dequeue and the amount larger
	#   determines the maximum we will prefetch.  Recommendation:
	#   min_after_dequeue + (num_threads + a small safety margin) * batch_size
	min_after_dequeue = batch_size
	capacity = min_after_dequeue + 3 * batch_size
	example_batch, label_batch = tf.train.shuffle_batch(
		[example, label], batch_size=batch_size, capacity=capacity,
		min_after_dequeue=min_after_dequeue)
	return example_batch, label_batch

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_nx1(x, n):
	return tf.nn.max_pool(x, ksize=[1, n, 1, 1],
		strides=[1, n, 1, 1], padding='SAME')

def main(_):

	filenames = ['shuffle_for_CNN.csv']
	seq_batch, label_batch = input_pipeline(filenames, 100)

	#placeholder, to be replaced by data input and label
	x = tf.placeholder(tf.float32, [None, 4 * 1024])
	y_ = tf.placeholder(tf.float32, shape=[None, 198])

	#Reshape the images
	x_image = tf.reshape(x, [-1, 1024, 4, 1])

	#Start network construction
	W_conv1 = weight_variable([first, 4, 1, 32])
	b_conv1 = bias_variable([32])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_nx1(h_conv1, first)


	W_conv2 = weight_variable([second, 4, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_nx1(h_conv2, second)


	h_pool2_flat = tf.reshape(h_pool2, [-1, int(1024 / (first * second) * 4 * 128)])


	W_fc1 = weight_variable([int(1024 / (first * second) * 4 * 128), fc])
	b_fc1 = bias_variable([fc])

	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	#Dropout, to reduce overfitting
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


	W_fc2 = weight_variable([fc, 198])
	b_fc2 = bias_variable([198])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	#Network construction ended


	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.InteractiveSession()
	coord = tf.train.Coordinator()
	
	#Training started
	#810000 records
	tf.initialize_all_variables().run()
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)
	for i in range(steps):
	#read data
		sequences, labels = sess.run([seq_batch, label_batch])
		sequences = [map(float, list(word)) for word in sequences]
		temp = []
		for n in labels:
			a = [0]*198
			a[n] = 1
			temp.append(a)
		labels = temp

		if i % 100 == 0:
			train_accuracy = accuracy.eval(feed_dict={
				x:sequences, y_: labels, keep_prob: 1.0})
			print("step %d, training accuracy %g"%(i, train_accuracy))

		sess.run(train_step, feed_dict={x: sequences, y_: labels, keep_prob: 0.5})

	# Test trained model
	sequences, labels = sess.run([seq_batch, label_batch])
	sequences = [map(float, list(word)) for word in sequences]
	temp = []
	for n in labels:
		a = [0]*198
		a[n] = 1
		temp.append(a)
	labels = temp

	print('Test result:', sess.run(accuracy, feed_dict={x: sequences,
		y_: labels, keep_prob: 1.0}))

	#stop coordinator
	coord.request_stop()
	coord.join(threads)

script, first, second, fc, steps = argv
first = int(first)
second = int(second)
fc = int(fc)
steps = int(steps)
#Serve as main function like in C/C++
if __name__ == '__main__':

	#It seems that this function would call main()
	tf.app.run()
