import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import paddle
from paddle.io import Dataset
from paddle.vision.transforms import transforms
from paddle.vision.models import resnet18
from paddle.nn import functional as F
print(paddle.__version__)

device = paddle.set_device('cpu')

Train_Dir = 'data/data60//training.csv'
Test_Dir = 'data/data60//test.csv'
lookid_dir = 'data/data60/IdLookupTable.csv'


class ImgTransforms(object):


    def __init__(self, fmt):
        self.format = fmt

    def __call__(self, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        img = img.transpose(self.format)

        if img.shape[0] == 1:
            img = np.repeat(img, 3, axis=0)
        return img


class FaceDataset(Dataset):
    def __init__(self, data_path, mode='train', val_split=0.2):
        self.mode = mode
        assert self.mode in ['train', 'val', 'test'], \
            "mode should be 'train' or 'test', but got {}".format(self.mode)
        self.data_source = pd.read_csv(data_path)

        self.data_source.dropna(how="any", inplace=True)
        self.data_label_all = self.data_source.drop('Image', axis=1)

        # 划分训练集和验证集合
        if self.mode in ['train', 'val']:
            np.random.seed(43)
            data_len = len(self.data_source)
            # 随机划分
            shuffled_indices = np.random.permutation(data_len)
            # 顺序划分
            # shuffled_indices = np.arange(data_len)
            self.shuffled_indices = shuffled_indices
            val_set_size = int(data_len * val_split)
            if self.mode == 'val':
                val_indices = shuffled_indices[:val_set_size]
                self.data_img = self.data_source.reindex().iloc[val_indices]
                self.data_label = self.data_label_all.reindex().iloc[val_indices]
            elif self.mode == 'train':
                train_indices = shuffled_indices[val_set_size:]
                self.data_img = self.data_source.reindex().iloc[train_indices]
                self.data_label = self.data_label_all.reindex().iloc[train_indices]
        elif self.mode == 'test':
            self.data_img = self.data_source
            self.data_label = self.data_label_all

        self.transforms = transforms.Compose([
            ImgTransforms((2, 0, 1))
        ])

    # 每次迭代时返回数据和对应的标签
    def __getitem__(self, idx):

        img = self.data_img['Image'].iloc[idx].split(' ')
        img = ['0' if x == '' else x for x in img]
        img = np.array(img, dtype='float32').reshape(96, 96)
        img = self.transforms(img)
        label = np.array(self.data_label.iloc[idx, :], dtype='float32') / 96
        return img, label

    # 返回整个数据集的总数
    def __len__(self):
        return len(self.data_img)


# 训练数据集和验证数据集
train_dataset = FaceDataset(Train_Dir, mode='train')
val_dataset = FaceDataset(Train_Dir, mode='val')

# 测试数据集
test_dataset = FaceDataset(Test_Dir, mode='test')
class FaceNet(paddle.nn.Layer):
    def __init__(self, num_keypoints, pretrained=False):
        super(FaceNet, self).__init__()
        self.backbone = resnet18(pretrained)
        self.outLayer1 = paddle.nn.Sequential(
            paddle.nn.Linear(1000, 512),
            paddle.nn.ReLU(),
            paddle.nn.Dropout(0.1))
        self.outLayer2 = paddle.nn.Linear(512, num_keypoints*2)
    def forward(self, inputs):
        out = self.backbone(inputs)
        out = self.outLayer1(out)
        out = self.outLayer2(out)
        return out
from paddle.static import InputSpec

# paddle.disable_static()
num_keypoints = 15
model = paddle.Model(FaceNet(num_keypoints))
optim = paddle.optimizer.Adam(learning_rate=1e-3,
    parameters=model.parameters())
model.prepare(optim, paddle.nn.MSELoss())
model.load('checkpoint')

result = model.predict(test_dataset, batch_size=1)

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2], y[1::2], marker='x', s=10, color='b')

fig = plt.figure(figsize=(10, 7))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
    axis = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    idx = np.random.randint(test_dataset.__len__())
    img, _ = test_dataset[idx]
    label_pred = result[0][idx].reshape(-1)
    label_pred = label_pred*96
    plot_sample(img[0], label_pred, axis)
plt.show()