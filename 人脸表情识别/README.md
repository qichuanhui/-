# 人脸表情识别
数据集是ferplus 请在kaggle上自行下载 并使用data_process.py进行处理后合并  
dlib库需要自行安装 并下载其dat文件  
实现了简单的人脸表情识别功能  
使用简单卷积神经网络，网络结构为： 
Conv -> ReLU -> Conv -> Relu -> Max Pooling -> Conv -> Relu -> Conv -> Relu ->
-> Max Pooling -> Conv -> ReLU -> Conv -> Relu ->Max Pooling ->
FC1(128) -> ReLU -> Dropout -> Affine -> Softmax  
数据集：ferplus  
模型测试集：电脑摄像头抓拍  
准确率：77%  
