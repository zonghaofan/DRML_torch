#### 

### 基于DRML改进的人脸表情分类

#### 一.train

#### 1.针对fer2013设计了一个网格划分的网络

执行python main_fer2013.py 即可

目前设计了四种方案的网络，只需要在main_fer2013.py里修改

from network_fer2013××  import Network_new

其中network_fer2013×× 有 四个.py 分别对应四种网络



其中下述的每个Region包含 conv、网络划分卷积、pool

方案1 network_fer2013.py

input(48x48)->Region1(4x4)->Region 2(4x4)->Region(4x1)->Bottleneck->Pool->Conv->AvgPool->relu->classifier

model path='./models_fer2013_aug_48*48/checkpoint.pth'



测试结果：

fer2013 private test acc:0.6746

CK acc：0.9225

jaffe acc:0.691

 

方案2   network_fer2013_deeper.py

input(96*96)->Region1(4x4)->Bottleneck(两次)->Region2(4x4)->Bottleneck(两次)->Region3(4x4)->Bottleneck(两次)->Region4(4x1)->conv->aver pool->classfier

inference效果很差



方案3 network_fer2013_deep_short.py

在每个region 加个short cut 形式 其余与方案1类似

model path='./models_fer2013_aug_deepsort/model.pth'



测试结果：

fer2013 private test acc:0.6801

CK acc：0.895

jaffe acc:0.6872



方案4 network_fer2013_deepest.py 延续方案3的short cut机制 只不过在两个region之间加入两个bottleneck,同时region区域改为2x2.

但是效果不理想。

model path='./models_fer2013_aug_deepest_48/model.pth'

测试结果：

fer2013 private test acc:0.59

CK acc: 0.78

方案5 在方案3的基础上，输入三通道，一个灰度图，一个canny提取轮廓图，一个细节增强的图
效果很差

方案6 DRML原始网络，输入三个增强后的通道，inference acc=0.56

#### 下一步方案：

测试结果可以看出，region之间加入 short cut 是有用的，两个region之间加入boottleneck反而不好，下一步针对网络结构进行修改，两个region之间不加入bottleneck，也可尝试用可分离卷积代替普通卷积。

2.先用dlib检测出人脸，根据眼睛、鼻子、下颚等去划分网络比例，这个的话batchsize就只能为1.



#### 2.针对OMG设计了一个网格划分的网络

训练执行python main_omg.py 即可

训练采用的是上述方案1。

由于OMG是提供的视频，这里用dlib制作了人脸数据集，具体制作细节，在另一个文件OMGEmotionChallenge的Readme，所以输入是RGB彩色图。

数据集path './data/OMG/OMG_train_data.h5' 

'./data/OMG/OMG_val_data.h5' 

model path='./models_omg/checkpoint.pth'

验证集测试结果为：0.925。



#### 二.inference

inference.py里面包含ck,fer2013,omg的inference

inference 里面是先将图片采用高斯滤波，在相减的方式获取细节增强后在进行推断，同时是用的5张照片，然后求概率平均，前者目前提高0.7%，后者提高0.2%。



#### 三.数据集

#### 1.各个数据集的表情对应的标签

OMG:     0-Anger 1-Disgust 2-Fear 3-Happy 4-Neutral 5-Sad 6-Surprise

fer2013: 0-Anger 1-Disgust 2-Fear 3-Happy 4-Sad 5-Surprise 6-Neutral

CK:0-Anger 1-Disgust 2-Fear 3-Happy 4-Sad 5-Surprise 6-Neutral

Jaffe:      0-Anger 1-Disgust 2-Fear 3-Happy 4-Sad 5-Surprise  6-Neutral



#### 2.fer2013数据集制作

由于fer2013的数据集在disgust的数据量过少,大概相差10倍  通过增加了omg的数据集达到了2169张，train数据集为：Anger：4333， Disgust：2169，Fear：4517， Happy：5000， Sad：5000，Surprise：3760， Neutral：5000.

test集是原先的private test.

生成的新数据集为fer2013_new_data.h5

 path：./data/fer2013/data/fer2013_new_data.h5



#### 3.其他数据集制作

OMG的train数据集为：OMG_train_data.h5，每一类是4,900.  

OMG的val数据集为：OMG_val_data.h5，每一类是500. 

path：./data/OMG/OMG_train_data.h5

​          ./data/OMG/OMG_val_data.h5

而CK和Jaffe的数据过少，制作测试用

CK path: ./data/CK/CK_data.h5

Jaffe path: ./data/jaffe

#### 环境：python3.5,pytorch1.0 
