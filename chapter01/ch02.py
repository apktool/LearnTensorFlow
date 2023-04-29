import tensorflow as tf


def get_w(epoch=40):
    lr = 0.2

    w = tf.Variable(tf.constant(5, dtype=tf.float32))
    for epoch in range(epoch):
        with tf.GradientTape() as tape:
            loss = tf.square(w + 1)
        grads = tape.gradient(loss, w)

        # w = w - lr*grads
        w.assign_sub(lr * grads)
        print("After %s epoch, w is %f, loss is %f" % (epoch, w.numpy(), loss))
