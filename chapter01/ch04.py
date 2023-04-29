import numpy as np
import tensorflow as tf


def gradient():
    with tf.GradientTape() as tape:
        x = tf.Variable(tf.constant(3.0))
        y = tf.pow(x, 2)
    grad = tape.gradient(y, x)
    print(grad)


def enumerate1():
    seq = ['one', 'two', 'three']
    for i, element in enumerate(seq):
        print(i, element)


def onehot():
    classes = 3
    labels = tf.constant([1, 0, 2])
    output = tf.one_hot(labels, depth=classes)
    print("result of labels1:", output)


def softmax():
    x1 = tf.constant([[5.8, 4.0, 1.2, 0.2]])  # 5.8,4.0,1.2,0.2（0）
    w1 = tf.constant([[-0.8, -0.34, -1.4],
                      [0.6, 1.3, 0.25],
                      [0.5, 1.45, 0.9],
                      [0.65, 0.7, -1.2]])
    b1 = tf.constant([2.52, -3.1, 5.62])
    y = tf.matmul(x1, w1) + b1
    print("x1.shape:", x1.shape)
    print("w1.shape:", w1.shape)
    print("b1.shape:", b1.shape)
    print("y.shape:", y.shape)
    print("y:", y)

    y_dim = tf.squeeze(y)

    y_pro = tf.nn.softmax(y_dim)
    print("y_dim:", y_dim)
    print("y_pro:", y_pro)


def assignsub():
    x = tf.Variable(4)
    x.assign_sub(1)
    print("x:", x)


def argmax():
    test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
    print("test:\n", test)
    print("每一列的最大值的索引：", tf.argmax(test, axis=0))
    print("每一行的最大值的索引", tf.argmax(test, axis=1))
