from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def read_my_file_format(filename_queue):
  reader = tf.TextLineReader()
  key, value = reader.read(filename_queue)
  record_defaults = [[0], [''], ['0'*1900]]
  col1, col2, col3 = tf.decode_csv(value, record_defaults=record_defaults)
  return col3, col1


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




def main(_):
  #set file reading operations
  filenames = ['shuffle_for_SOFTMAX.csv']
  seq_batch, label_batch = input_pipeline(filenames, 100)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 1900])
  W = tf.Variable(tf.zeros([1900, 198]))
  b = tf.Variable(tf.zeros([198]))
  y = tf.matmul(x, W) + b

  #place to hold the labels
  y_ = tf.placeholder(tf.float32, [None, 198])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))  #like error
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)       #train to reduce error
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))                #check if each predict match the corresponding label, and
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))              #calculate the proportion of correct predictions over all predictions, named accuracy
  
  sess = tf.InteractiveSession()

  #must-do procedure for reading files
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord, sess=sess)

  # Train
  tf.initialize_all_variables().run()
  for i in range(5000):
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
        x:sequences, y_: labels})
      print("step %d, training accuracy %g"%(i, train_accuracy))

    sess.run(train_step, feed_dict={x: sequences, y_: labels})

  # Test trained model
  c = 0
  for n in range(1000):
    sequences, labels = sess.run([seq_batch, label_batch])
    sequences = [map(float, list(word)) for word in sequences]
    temp = []
    for n in labels:
      a = [0]*198
      a[n] = 1
      temp.append(a)
    labels = temp
    c +=  sess.run(accuracy, feed_dict={x: sequences, y_: labels}) * 100

  print('Test result:', c / 100000.0)

  #stop coordinator
  coord.request_stop()
  coord.join(threads)

if __name__ == '__main__':
  tf.app.run()      #act like main()?

