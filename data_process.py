from keras.preprocessing.image import ImageDataGenerator
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
# tf.compat.v1.disable_eager_execution()

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

LABEL_DICT = {
    'anger': 0, 'contempt': 1, 'disgust': 2, 'fear': 3, 'happiness': 4, 'neutral': 5, 'sadness': 6, 'surprise': 7
}
# 数据预处理
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')
# ./train 是数据文件夹的父文件夹 为了方便 每次只在该文件夹内留下一个类别
# 在./facialtrain下建立和./train下相同的文件目录 每次改变./train 文件夹下留下的分类 并修改save_to_dir为对应名字文件夹调用此函数
# 最后将处理好的数据和原来的数据合并即可
gener = datagen.flow_from_directory(r'C:\Users\77191\Desktop\opencv_ml\data\train',
                                    target_size=(48, 48),
                                    batch_size=1,
                                    shuffle=False,
                                    save_to_dir=r'C:\Users\77191\Desktop\opencv_ml\data\facialtrain\surprise',
                                    save_prefix='trans_',
                                    save_format='png',
                                    )
for i in range(gener.samples):
    gener.next()
