"""
: author:Blues Traveller
: module_name:emotion_classifier
: purpose:实现简单的人脸表情识别（不包括GUI）
: date:2022/7/15

使用简单卷积神经网络，网络结构为：
Conv -> ReLU -> Conv -> Relu -> Max Pooling -> Conv -> Relu -> Conv -> Relu ->
-> Max Pooling -> Conv -> ReLU -> Conv -> Relu ->Max Pooling ->
FC1(128) -> ReLU -> Dropout -> Affine -> Softmax

数据集：ferplus
模型测试集：电脑摄像头抓拍
准确率：77%
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

MODEL_PATH = "./model/cnn_emotion/emotion.ckpt"
TRAIN_DIR = "./data/facialtrain"
TEST_DIR = "./data/test"
# 汉字图片重置的宽、高
IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48
CLASSIFICATION_COUNT = 8
LABEL_DICT = {
    'anger': 0, 'contempt': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'neutral': 5, 'sadness': 6, 'surprise': 7
}
# 学习率
learning_rate = 1e-3
# 循环次数
max_epochs = 42
# 批次大小
batch_size = 32
# 检查点步长
check_step = 100


def load_data(dir_path):
    data = []
    labels = []
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isdir(item_path):
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                gray_image = cv.imread(subitem_path, cv.IMREAD_GRAYSCALE)
                resized_image = cv.resize(gray_image, (IMAGE_WIDTH, IMAGE_HEIGHT))
                # 去噪（非局部平均去噪 耗时长 效果较好 但是经测试不会提高正确率）
                # resized_image = cv.fastNlMeansDenoising(resized_image, templateWindowSize=7, searchWindowSize=21, h=5)
                data.append(resized_image.ravel())
                labels.append(LABEL_DICT[item])

    return np.array(data), np.array(labels)


# 正规化
def normalize_data(data):
    return (data - data.mean()) / data.max()


# 独热编码
def onehot_labels(labels):
    onehots = np.zeros((len(labels), CLASSIFICATION_COUNT))
    for i in np.arange(len(labels)):
        onehots[i, labels[i]] = 1
    return onehots


# 初始化系数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 二维卷积神经网络（层）
def conv2d(x, W):
    # padding='SAME',使卷积输出的尺寸=ceil(输入尺寸/stride)，必要时自动padding
    # padding='VALID',不会自动padding，对于输入图像右边和下边多余的元素，直接丢弃
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 最大池化 2*2
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT * IMAGE_WIDTH])
y_ = tf.placeholder(tf.float32, shape=[None, CLASSIFICATION_COUNT])
x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

W_conv1 = weight_variable([3, 3, 1, 32])  # color channel == 1; 32 filters
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 48*48

W_conv2 = weight_variable([3, 3, 32, 32])
b_conv2 = bias_variable([32])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)  # 48*48
h_pool2 = max_pool_2x2(h_conv2)  # 48x48 => 24x24

W_conv3 = weight_variable([3, 3, 32, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)  # 24*24

W_conv4 = weight_variable([3, 3, 64, 64])
b_conv4 = bias_variable([64])
h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
h_pool3 = max_pool_2x2(h_conv4)  # 24*24 => 12*12

W_conv5 = weight_variable([3, 3, 64, 128])
b_conv5 = bias_variable([128])
h_conv5 = tf.nn.relu(conv2d(h_pool3, W_conv5) + b_conv5)  # 12*12

W_conv6 = weight_variable([3, 3, 128, 128])
b_conv6 = bias_variable([128])
h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)
h_pool4 = max_pool_2x2(h_conv6)  # 12*12 => 6*6

# 全连接神经网络的第一个隐藏层
# 池化层输出的元素总数为：6(H)*6(W)*128(filters)
W_fc1 = weight_variable([6 * 6 * 128, 128])  # 全连接第一个隐藏层神经元128个
b_fc1 = bias_variable([128])
h_pool3_flat = tf.reshape(h_pool4, [-1, 6 * 6 * 128])  # 转成1列
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)  # Affine+ReLU

keep_prob = tf.placeholder(tf.float32)  # 定义Dropout的比例
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 执行dropout

# 全连接神经网络输出层
W_fc2 = weight_variable([128, CLASSIFICATION_COUNT])  # 全连接输出为 CLASSIFICATION_COUNT 个
b_fc2 = bias_variable([CLASSIFICATION_COUNT])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 使用softmax成本函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 计数测试点循环次数
count_epochs = []
# 记录测试准确率
Accuracy = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print("装载训练数据...")
    train_data, train_labels = load_data(TRAIN_DIR)
    train_data = normalize_data(train_data)
    train_labels = onehot_labels(train_labels)
    print("装载%d条数据，每条数据%d个特征" % (train_data.shape))

    train_samples_count = len(train_data)
    train_indicies = np.arange(train_samples_count)
    # 获得打乱的索引序列
    np.random.shuffle(train_indicies)

    print("装载测试数据...")
    test_data, test_labels = load_data(TEST_DIR)
    test_data = normalize_data(test_data)
    test_labels = onehot_labels(test_labels)
    print("装载%d条数据，每条数据%d个特征" % (test_data.shape))

    iters = int(np.ceil(train_samples_count / batch_size))
    print("Training...")
    for epoch in range(1, max_epochs + 1):
        print("Epoch #", epoch)
        for i in range(1, iters + 1):
            # 获取本批数据
            start_idx = (i * batch_size) % train_samples_count
            idx = train_indicies[start_idx: start_idx + batch_size]
            batch_x = train_data[idx, :]
            batch_y = train_labels[idx, :]
            _, batch_accuracy = sess.run([train_step, accuracy], feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.6})
            if i % check_step == 0:
                print("Iter:", i, "of", iters, "batch_accuracy=", batch_accuracy)

        if epoch % 2 == 0:
            count_epochs.append(epoch)
            test_accuracy = accuracy.eval(feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0})
            Accuracy.append(test_accuracy)
            print('Test accuracy %g' % test_accuracy)
    print("Training completed.")

    print("Saving model...")
    saver = tf.train.Saver()
    saved_file = saver.save(sess, MODEL_PATH)
    print('Model saved to ', saved_file)

    test_accuracy = accuracy.eval(feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0})
    print('Test accuracy %g' % test_accuracy)  # 约0.773

    # 可视化epochs 和 accuracy曲线，便于选择最佳epochs
    plt.plot(count_epochs, Accuracy, ls=":", color="k", marker="o", lw=2, label="epochs-accuracy")
    plt.xlabel('Test_Accuracy')
    plt.ylabel('Epochs')
    plt.title('Epochs-Accuracy')
    plt.legend()
    plt.show()
