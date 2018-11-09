import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import dataset.MNIST.mnist as mnist
import model.LogisticRegression as LR


print("start download minst")
mnist_data_sets = mnist.read_data_sets("../data/mnist", one_hot=True)
print("start lr")

lr = LR.LogisticRegression()
lr.build_model()
batch_xs,batch_ys = mnist_data_sets.train.next_batch(55000)
lr.train(batch_xs, batch_ys)
batch_xs,batch_ys = mnist_data_sets.test.images,mnist_data_sets.test.labels
lr.test(batch_xs,batch_ys)