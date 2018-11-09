import tensorflow as tf
import numpy as np





input =[[1],[2],[3],[4],[5]]
input =  np.array(input)

x = tf.placeholder("float")
y_ = tf.placeholder("float")
W = tf.Variable(np.random.rand(1,1), name="weight", dtype=tf.float32)
b = tf.Variable(tf.ones([1,1]), name="bias", dtype=tf.float32)
y = tf.add(tf.matmul(x, W), b)


# x = tf.constant(
# [[1., 1.],
# [2., 2.]])

opt =  tf.reduce_mean(x)  # 1.5

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
result_y =  sess.run(y,feed_dict={x:input})
print(result_y)



## loss 损失函数
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)), reduction_indices=1)
#train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
