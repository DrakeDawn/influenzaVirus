#Some must-use packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy

from sys import argv
import tensorflow as tf

CLASSES = 198
TRAINING_RECORDS = 100
VALIDATION_RECORDS = 10000
QSIZE = 12

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
	filenames = ['shuffle_CNN_training.csv']
	seq_batch, label_batch = input_pipeline(filenames, TRAINING_RECORDS)
	fnames_validation = ['shuffle_CNN_validation.csv']
	seq_batch_v, label_batch_v = input_pipeline(fnames_validation, VALIDATION_RECORDS)

	#placeholder, to be replaced by data input and label
	x = tf.placeholder(tf.float32, [None, 4 * 1024])
	y_ = tf.placeholder(tf.float32, shape=[None, CLASSES])

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


	h_pool2_flat = tf.reshape(h_pool2, [-1, int(1024 / (first * second) * 4 * 64)])


	W_fc1 = weight_variable([int(1024 / (first * second) * 4 * 64), fc])
	b_fc1 = bias_variable([fc])

	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	#Dropout, to reduce overfitting
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


	W_fc2 = weight_variable([fc, CLASSES])
	b_fc2 = bias_variable([CLASSES])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	#Network construction ended


	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	sess = tf.InteractiveSession()
	coord = tf.train.Coordinator()

	tf.initialize_all_variables().run()
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)

	seq_validation, label_validation = sess.run([seq_batch_v, label_batch_v])
	seq_validation = [map(float, list(word)) for word in seq_validation]
	temp = []
	for n in label_validation:
		a = [0]*CLASSES
		a[n] = 1
		temp.append(a)
	label_validation = temp
	q = []

	#Training started
	#610000 records
	for i in range(steps):
	#read & parse data
		sequences, labels = sess.run([seq_batch, label_batch])
		sequences = [map(float, list(word)) for word in sequences]
		temp = []
		for n in labels:
			a = [0]*CLASSES
			a[n] = 1
			temp.append(a)
		labels = temp

		if len(q) == QSIZE:
			q = q[1:]
		q.append(accuracy.eval(feed_dict=
			{x: seq_validation, y_: label_validation, keep_prob: 1.0}))
		narray1 = numpy.array(q)
		sum1 = narray1.sum()
		narray2 = narray1 * narray1
		sum2 = narray2.sum()
		mean = sum1 / len(q)
		var = sum2 / len(q) - mean ** 2
		print("Validation variance:", var)

		#training accuracy
		if i % 100 == 0:
			train_accuracy = accuracy.eval(feed_dict={
				x: sequences, y_: labels, keep_prob: 1.0})
			print("step %d, training accuracy %g"%(i, train_accuracy))
		#train
		sess.run(train_step, feed_dict={x: sequences, y_: labels, keep_prob: 0.5})

	# Test trained model
	test_result = 0
	for j in range(100):
		sequences[:] = []
		labels[:] = []
		for i in range(10):
			temps, templ = sess.run([seq_batch, label_batch])
			sequences.extend(temps)
			labels.extend(templ)
		sequences = [map(float, list(word)) for word in sequences]
		temp = []
		for n in labels:
			a = [0]*CLASSES
			a[n] = 1
			temp.append(a)
		labels = temp
		test_result += int(sess.run(accuracy, feed_dict={x: sequences, y_: labels, keep_prob: 1.0}) * 1000)

	test_result = test_result / 100000.0
	print('Test result:', test_result)

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
