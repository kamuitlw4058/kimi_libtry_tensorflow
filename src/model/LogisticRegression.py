import tensorflow as tf
import numpy as np


class LogisticRegression:

    def  __init__(self):
        self.learning_rate = 0.01  # 学习率
        self.x = None
        self.y_ = None
        self.W = None
        self.b = None
        self.y = None
        self.cross_entropy = None
        self.learning_rate = 0.01  # 学习率
        self.train_step = None
        self.sess = None
        return


    def build_model(self):
        self.x = tf.placeholder("float")
        self.y_ = tf.placeholder("float")
        self.W = tf.Variable(np.random.randn(), name="weight", dtype=tf.float32)
        self.b = tf.Variable(np.random.randn(), name="bias", dtype=tf.float32)
        self.y = tf.sigmoid(tf.add(tf.matmul(self.x, self.W), self.b))
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y)), reduction_indices=1)
        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy)


    def train(self,train_x,train_y):

        if self.sess == None :
            init = tf.initialize_all_variables()
            self.sess = tf.Session()
            self.sess.run(init)
        for (x, y) in zip(train_x, train_y):
            self.sess.run(self.train_step, feed_dict={self.x: x, self.y_: y})

    def test(self,test_x,test_y):
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print ("accuracy:" + self.sess.run(accuracy, feed_dict={self.x: test_x, self.y_: test_y}))

    def prediction(self,x):
        result = self.sess.run(self.y, feed_dict={x: x})  # result是一个向量，通过索引来判断图片数字
        



