import tensorflow as tf

def read_my_file_format(filename_queue):
	reader = tf.TextLineReader()
	key, value = reader.read(filename_queue)
	record_defaults = [[0], [''], ['']]
	col1, col2, col3 = tf.decode_csv(value, record_defaults=record_defaults)
	return col3, col1



def input_pipeline(batch_size, read_threads, num_epochs=None):
	filename_queue = tf.train.string_input_producer(["0.csv", "1.csv", "2.csv", "3.csv", "4.csv"])
	examplelist = read_my_file_format(filename_queue)
	# min_after_dequeue defines how big a buffer we will randomly sample
	#   from -- bigger means better shuffling but slower start up and more
	#   memory used.
	# capacity must be larger than min_after_dequeue and the amount larger
	#   determines the maximum we will prefetch.  Recommendation:
	#   min_after_dequeue + (num_threads + a small safety margin) * batch_size
	min_after_dequeue = 10
	capacity = min_after_dequeue + 6 * batch_size
	example_batch, label_batch = tf.train.shuffle_batch(
		examplelist, batch_size=batch_size, capacity=capacity,
		min_after_dequeue=min_after_dequeue, num_threads=read_threads)
	return example_batch, label_batch

