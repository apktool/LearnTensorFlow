import numpy as np
import tensorflow as tf


def constant():
    a = tf.constant(1, dtype=tf.int64)
    print("a: ", a)
    print("a.dtype:", a.dtype)
    print("a.shape:", a.shape)

    b = tf.constant([1, 2], dtype=tf.int64)
    print("b: ", b)
    print("b.dtype:", b.dtype)
    print("b.shape:", b.shape)


def tensor():
    a = np.arange(0, 5)
    b = tf.convert_to_tensor(a, dtype=tf.int64)
    print(a)
    print(b)

    c = tf.zeros([2, 3], dtype=tf.int64)
    print(c)

    d = tf.ones(4, dtype=tf.int64)
    print(d)

    e = tf.fill([2, 3], 9)
    print(e)

    # 2x2 以 0.5 为均值，1 为标准差的分布
    f = tf.random.normal([2, 2], mean=0.5, stddev=1)
    print(f)

    # 2x2 以 0.5 为均值，1 为标准差的分布，数据更向 0.5 集中
    g = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)
    print(g)
