import tensorflow as tf
import numpy as np
from helper import get_data
class VGG():
    def __init__(self):
        self.x_train, self.y_train, self.x_val, self.y_val = get_data()
        self.save_path = './model/'


    def build_network(self, height = 224, width = 224, channel = 3):
        self.x = tf.placeholder(tf.float32, shape = [None, height, width, channel], name = 'input')
        self.y = tf.placeholder(tf.int64, shape = [None, 2], name = 'label')
        with tf.name_scope('conv1_1') as scope:
            kernel = self.weight_variable([3, 3, 3, 64])
            bias = self.bias_variable([64])
            output_conv1_1 = tf.nn.relu(self.conv2d(self.x, kernel) + bias, name = scope)

        with tf.name_scope('conv1_2') as scope:
            kernel = self.weight_variable([3, 3, 64, 64])
            bias = self.bias_variable([64])
            output_conv1_2 = tf.nn.relu(self.conv2d(output_conv1_1, kernel) + bias, name = scope)

        pool1 = self.pool_max(output_conv1_2)

        with tf.name_scope('conv2_1') as scope:
            kernel = self.weight_variable([3, 3, 64, 128])
            bias = self.bias_variable([128])
            output_conv2_1 = tf.nn.relu(self.conv2d(pool1, kernel) + bias, name = scope)


        with tf.name_scope('conv2_2') as scope:
            kernel = self.weight_variable([3, 3, 128, 128])
            bias = self.bias_variable([128])
            output_conv2_2 = tf.nn.relu(self.conv2d(output_conv2_1, kernel) + bias, name = scope)

        pool2 = self.pool_max(output_conv2_2)

        
        with tf.name_scope('conv3_1') as scope:
            kernel = self.weight_variable([3, 3, 128, 256])
            bias = self.bias_variable([256])
            output_conv3_1 = tf.nn.relu(self.conv2d(pool2, kernel) + bias, name = scope)

        
        with tf.name_scope('conv3_2') as scope:
            kernel = self.weight_variable([3, 3, 256, 256])
            bias = self.bias_variable([256])
            output_conv3_2 = tf.nn.relu(self.conv2d(output_conv3_1, kernel) + bias, name = scope)

        with tf.name_scope('conv3_3') as scope:
            kernel = self.weight_variable([3, 3, 256, 256])
            biases = self.bias_variable([256])
            output_conv3_3 = tf.nn.relu(self.conv2d(output_conv3_2, kernel) + biases, name=scope)

        pool3 = self.pool_max(output_conv3_3)

        with tf.name_scope('conv4_1') as scope:
            kernel = self.weight_variable([3, 3, 256, 512])
            bias = self.bias_variable([512])
            output_conv4_1 = tf.nn.relu(self.conv2d(pool3, kernel) + bias, name = scope)

        
        with tf.name_scope('conv4_2') as scope:
            kernel = self.weight_variable([3, 3, 512, 512])
            bias = self.bias_variable([512])
            output_conv4_2 = tf.nn.relu(self.conv2d(output_conv4_1, kernel) + bias, name = scope)
        
        with tf.name_scope('conv4_3') as scope:
            kernel = self.weight_variable([3, 3, 512, 512])
            biases = self.bias_variable([512])
            output_conv4_3 = tf.nn.relu(self.conv2d(output_conv4_2, kernel) + biases, name=scope)

        pool4 = self.pool_max(output_conv4_3)


        with tf.name_scope('conv5_1') as scope:
            kernel = self.weight_variable([3, 3, 512, 512])
            bias = self.bias_variable([512])
            output_conv5_1 = tf.nn.relu(self.conv2d(pool4, kernel) + bias, name = scope)

        
        with tf.name_scope('conv5_2') as scope:
            kernel = self.weight_variable([3, 3, 512, 512])
            bias = self.bias_variable([512])
            output_conv5_2 = tf.nn.relu(self.conv2d(output_conv5_1, kernel) + bias, name = scope)

        with tf.name_scope('conv5_3') as scope:
            kernel = self.weight_variable([3, 3, 512, 512])
            biases = self.bias_variable([512])
            output_conv5_3 = tf.nn.relu(self.conv2d(output_conv5_2, kernel) + biases, name=scope)

        pool5 = self.pool_max(output_conv5_3)

        #fc6
        with tf.name_scope('fc6') as scope:
            shape = int(np.prod(pool5.get_shape()[1:]))
            kernel = self.weight_variable([shape, 4096])
            biases = self.bias_variable([4096])
            pool5_flat = tf.reshape(pool5, [-1, shape])
            output_fc6 = tf.nn.relu(self.fc(pool5_flat, kernel, biases), name=scope)

        #fc7
        with tf.name_scope('fc7') as scope:
            kernel = self.weight_variable([4096, 4096])
            biases = self.bias_variable([4096])
            output_fc7 = tf.nn.relu(self.fc(output_fc6, kernel, biases), name=scope)

        #fc8
        with tf.name_scope('fc8') as scope:
            kernel = self.weight_variable([4096, 2])
            biases = self.bias_variable([2])
            output_fc8 = tf.nn.relu(self.fc(output_fc7, kernel, biases), name=scope)
        
        finaloutput = output_fc8

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=finaloutput, labels=self.y))
        optimize = tf.train.AdamOptimizer(learning_rate=0.00000005).minimize(cost)

        # prediction_labels = tf.argmax(finaloutput, axis=1, name="output")
        read_labels = self.y

        correct_prediction = tf.equal(tf.argmax(finaloutput, 1), tf.argmax(read_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # correct_times_in_batch = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))

        return dict(
            x=self.x,
            y=self.y,
            optimize=optimize,
            correct_prediction=correct_prediction,
            cost=cost,
            accuracy=accuracy,
            finaloutput=finaloutput
        )


    def train_network(self, graph, batch_size, num_epochs):
        init = tf.global_variables_initializer()
        Saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)            
            for epoch in range(num_epochs):
                for i in range(int(len(self.x_train) / batch_size) + 1):
                    batch_x = self.x_train[i * batch_size:(i+1) * batch_size]
                    batch_y = self.y_train[i * batch_size:(i+1) * batch_size]
                    _, cost, output = sess.run([graph['optimize'], graph['cost'], graph['finaloutput']], feed_dict={graph['x'] : batch_x, graph['y']: batch_y})
                    # print('Epoch: %d , batch loss: %f' % (epoch, cost))
                    # print(output)
                    # # print(batch_y)

                train_accuracy, train_loss = sess.run([graph['accuracy'], graph['cost']], feed_dict={graph['x'] : self.x_train, graph['y'] : self.y_train})
                test_accuracy, test_loss = sess.run([graph['accuracy'], graph['cost']], feed_dict={graph['x'] : self.x_val, graph['y']: self.y_val})
                print('Epoch: %d ,Train Accuracy: %0.4f ,Train loss: %0.4f ,Test Accuracy: %0.4f ,Test loss: %0.4f' % (epoch+1, train_accuracy, train_loss, test_accuracy, test_loss))
                if((epoch+1) % 10 == 0):
                    spath = Saver.save(sess, self.save_path)
                    print("Model save in file: %s" %spath)

    def weight_variable(self, shape, name = 'weight'):
        initial = tf.truncated_normal(shape, dtype = tf.float32, stddev = 0.1)
        return tf.Variable(initial, name = name)

    def bias_variable(self, shape, name = 'bias'):
        initial = tf.constant(0.1, dtype = tf.float32, shape = shape)
        return tf.Variable(initial, name = name)

    def pool_max(self, input):
        return tf.nn.max_pool(input,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

    def conv2d(self, input, w):
        return tf.nn.conv2d(input, w, [1,1,1,1], padding='SAME')
    
    def fc(self, input, w, b):
        return tf.matmul(input, w) + b