## keypoint_info/sigmas


### sigmas
用于计算OKS （Object Keypoint Similarity）分数。
提出于COCO关键点检测数据集reference: https://cocodataset.org/#keypoints-eval
OKS可以类比目标检测中的IoU，用于衡量检测结果的准确度。
OKS本质上是计算gt和预测关键点之间的距离，然后根据sigmas计算OKS分数。

### OKS
对于每个检测物体，有关键点集{x_1, y_1, v_1, ..., x_n, y_n, v_n}，其中(x_i, y_i)是关键点的坐标，v_i是关键点的可见性标志。
v = 0表示关键点为被标记，v = 1表示关键点被标记但是不可见，v = 2表示关键点被标记且可见。

每个gt物体还有一个尺度s，定义为分割面积的平方根。

OKS定义为：
OKS = \frac{\sum_{i} exp(-d_i^2 / 2 / \sigma_i^2)}{\sum_{i} {1:v_i>0}}
