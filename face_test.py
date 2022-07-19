import cv2 as cv
from face_split import FaceSpliter
import os
import tensorflow._api.v2.compat.v1 as tf
import time
from matplotlib import pyplot as plt
import numpy as np
import face_split

tf.disable_v2_behavior()

# 人脸识别检测器路径
path_to_lib = "./libs/dlib/shape_predictor_68_face_landmarks.dat"
# 读取未切割的图像的路径 以及摄像头获取的截图的储存路径
path_to_read = "./images/capture/"
# 用来存储生成的单张切割好的人脸的路径
path_to_save = "./images/split_face_test/"
# 训练好的模型路径
# path_to_model =
# 图片大小
IMAGE_SIZE = 48
IMAGE_WIDTH = 48
IMAGE_HEIGHT = 48
# 类别数目
CLASSIFICATION_COUNT = 8
# 标签
LABEL_DICT = {
    'anger': 0, 'contempt': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'neutral': 5, 'sadness': 6, 'surprise': 7
}


class TestResult:
    def __init__(self):
        self.anger = 0
        self.contempt = 0
        self.disgust = 0
        self.fear = 0
        self.happiness = 0
        self.neutral = 0
        self.sadness = 0
        self.surprise = 0

    def evaluate(self, label):
        if (0 == label):
            self.anger = self.anger + 1
        if (1 == label):
            self.contempt = self.contempt + 1
        if (2 == label):
            self.disgust = self.disgust + 1
        if (3 == label):
            self.fear = self.fear + 1
        if (4 == label):
            self.happiness = self.happiness + 1
        if (5 == label):
            self.neutral = self.neutral + 1
        if (6 == label):
            self.sadness = self.sadness + 1
        if (7 == label):
            self.surprise = self.surprise + 1

    def display_result(self, evaluations):
        print("生气 = " + str((self.anger / float(evaluations)) * 100) + "%")
        print("蔑视 = " + str((self.contempt / float(evaluations)) * 100) + "%")
        print("反感 = " + str((self.disgust / float(evaluations)) * 100) + "%")
        print("恐惧 = " + str((self.fear / float(evaluations)) * 100) + "%")
        print("高兴 = " + str((self.happiness / float(evaluations)) * 100) + "%")
        print("难过 = " + str((self.sadness / float(evaluations)) * 100) + "%")
        print("惊喜 = " + str((self.surprise / float(evaluations)) * 100) + "%")
        print("无表情 = " + str((self.neutral / float(evaluations)) * 100) + "%")


def load_data(dir_path):
    data = []
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        gray_image = cv.imread(item_path, cv.IMREAD_GRAYSCALE)
        resized_image = cv.resize(gray_image, (IMAGE_HEIGHT, IMAGE_WIDTH))
        plt.imshow(resized_image, cmap=plt.get_cmap('gray'))
        plt.show()
        normalized_image = normalize_data(resized_image)
        data.append(normalized_image.ravel())
    return np.array(data)


def normalize_data(data):
    return (data - data.mean()) / data.max()


def onehot_labels(labels):
    onehots = np.zeros((len(labels), CLASSIFICATION_COUNT))
    for i in np.arange(len(labels)):
        onehots[i, labels[i]] = 1
    return onehots


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # padding='SAME',使卷积输出的尺寸=ceil(输入尺寸/stride)，必要时自动padding
    # padding='VALID',不会自动padding，对于输入图像右边和下边多余的元素，直接丢弃
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, shape=[IMAGE_HEIGHT * IMAGE_WIDTH])
    y_ = tf.placeholder(tf.float32, shape=[None, CLASSIFICATION_COUNT])
    x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_HEIGHT, 1])

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
    h_pool4 = max_pool_2x2(h_conv6)  # 6*6

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

    EMOTION_MODEL_PATH = "./model/cnn_emotion/emotion.ckpt"
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, EMOTION_MODEL_PATH)

    digit_image = load_data(path_to_save)
    # 计数第几个样本
    count = 1
    # 每个样本测试次数
    num_evaluation = 1
    for image in digit_image:
        tResult = TestResult()
        print("测试用例%d开始训练" % count)
        count = count + 1
        start_time = time.time()
        for i in range(num_evaluation):
            results = sess.run(y_conv, feed_dict={x: image, keep_prob: 1.0})
            predict_label = np.argmax(results[0])
            tResult.evaluate(predict_label)
        end_time = time.time()
        tResult.display_result(num_evaluation)
        print("用时----> %s 秒" % (end_time - start_time))





