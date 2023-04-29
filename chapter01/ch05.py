import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn import datasets


def load():
    x_data = datasets.load_iris().data
    y_data = datasets.load_iris().target
    print("x_data from datasets: \n", x_data)
    print("y_data from datasets: \n", y_data)

    x_data = DataFrame(x_data, columns=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'])
    pd.set_option('display.unicode.east_asian_width', True)  # 设置列名对齐
    print("x_data add index: \n", x_data)

    x_data['类别'] = y_data
    print("x_data add a column: \n", x_data)


def train():
    x_data = datasets.load_iris().data
    y_data = datasets.load_iris().target
    np.random.seed(116)
    np.random.shuffle(x_data)
    np.random.seed(116)
    np.random.shuffle(y_data)
    tf.random.set_seed(116)

    # 训练集
    x_train = x_data[:-30]
    x_train = tf.cast(x_train, tf.float32)
    y_train = y_data[:-30]

    # 测试集
    x_test = x_data[-30:]
    x_test = tf.cast(x_test, tf.float32)
    y_test = y_data[-30:]

    train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

    w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
    b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

    lr = 0.1
    train_loss_results = []
    test_acc = []
    epoch = 500
    loss_all = 0

    # 训练模型
    for epoch in range(epoch):  # 数据集级别的循环，每个epoch循环一次数据集
        for step, (x_train, y_train) in enumerate(train_db):  # batch级别的循环 ，每个step循环一个batch
            with tf.GradientTape() as tape:  # with结构记录梯度信息
                y = tf.matmul(x_train, w1) + b1  # 神经网络乘加运算
                y = tf.nn.softmax(y)  # 使输出y符合概率分布（此操作后与独热码同量级，可相减求loss）
                y_ = tf.one_hot(y_train, depth=3)  # 将标签值转换为独热码格式，方便计算loss和accuracy
                loss = tf.reduce_mean(tf.square(y_ - y))  # 采用均方误差损失函数mse = mean(sum(y-out)^2)
                loss_all += loss.numpy()  # 将每个step计算出的loss累加，为后续求loss平均值提供数据，这样计算的loss更准确
            # 计算loss对各个参数的梯度
            grads = tape.gradient(loss, [w1, b1])

            # 实现梯度更新 w1 = w1 - lr * w1_grad      b = b - lr * b_grad
            w1.assign_sub(lr * grads[0])
            b1.assign_sub(lr * grads[1])

        # 每个epoch，打印loss信息
        print("Epoch {}, loss: {}".format(epoch, loss_all / 4))
        train_loss_results.append(loss_all / 4)  # 将4个step的loss求平均记录在此变量中
        loss_all = 0  # loss_all归零，为记录下一个epoch的loss做准备

        # 测试结果
        total_correct, total_number = 0, 0
        for x_test, y_test in test_db:
            # 使用更新后的参数进行预测
            y = tf.matmul(x_test, w1) + b1
            y = tf.nn.softmax(y)
            pred = tf.argmax(y, axis=1)  # 返回y中最大值的索引，即预测的分类
            # 将pred转换为y_test的数据类型
            pred = tf.cast(pred, dtype=y_test.dtype)
            # 若分类正确，则correct=1，否则为0，将bool型的结果转换为int型
            correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
            # 将每个batch的correct数加起来
            correct = tf.reduce_sum(correct)
            # 将所有batch中的correct数加起来
            total_correct += int(correct)
            # total_number为测试的总样本数，也就是x_test的行数，shape[0]返回变量的行数
            total_number += x_test.shape[0]
        # 总的准确率等于total_correct/total_number
        acc = total_correct / total_number
        test_acc.append(acc)
        print("Test_acc:", acc)
        print("--------------------------")

    # 画图
    plt.title('Loss Function Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(train_loss_results, label="$Loss$")
    plt.legend()
    plt.show()

    # 绘制 Accuracy 曲线
    plt.title('Acc Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.plot(test_acc, label="$Accuracy$")
    plt.legend()
    plt.show()
