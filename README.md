# 项目说明

本项目使用yolov3对目标在2D图像中的位置进行检测，获取检测框的4个顶点坐标。然后将坐标映射到2D图像对应的3D数据中，根据3D数据计算出物体的真实尺寸。

## 环境准备
win10.

torch 1.6.0.

python3.6
### 训练

#### 数据集准备

自己的数据集标记好后, png文件放入 `datasets/DOTA_data/images`中，xml文件放入`annotation`中。
然后运行 `data_generate_txtfile.py` 在`ImageSet`中生成`train.txt` `val.txt` `test.txt`三个文件，再运行`xml_txt.py`将`annotations`
中的xml文件转换成`txt`文件存在`labelTxt`中。


#### config

对`cfg/yolov3.cfg`的603、610、689、696、776、783行进行修改，1修改filters的数量，2修改classes的数量（即目标检测的类别数）. 

对`cfg/icdar.names`进行修改，写入标签的名称，名称顺序与`datasets/DOTA_data/xml_txt.py`中classes的名称顺序一致。
#### 训练网络

运行`train.py`训练网络

#### 预测模型

运行`predict.py`，修改自己训练好的网络权重位置和预测结果存放位置。

#### 预测和测量

运行`detect.py`对物体进行尺寸测量，测量时依赖[zivid](https://www.zivid.com/zivid-one-plus)
相机采到的数据。
