# Basic Pipeline with MPII

Author： hty

## 准备数据集

### **1. 下载MPII数据集**
    
图片文件： http://human-pose.mpi-inf.mpg.de/
    
标注文件： https://download.openmmlab.com/mmpose/datasets/mpii_annotations.tar


### **2. 数据集组织形式：**

参考：https://mmpose.readthedocs.io/en/latest/dataset_zoo/2d_body_keypoint.html


    mmpose
    ├── mmpose
    ├── docs
    ├── tests
    ├── tools
    ├── configs
    `── data
        │── mpii
            |── annotations
            |   |── mpii_gt_val.mat
            |   |── mpii_test.json
            |   |── mpii_train.json
            |   |── mpii_trainval.json
            |   `── mpii_val.json
            `── images
                |── 000001163.jpg
                |── 000003072.jpg

### **3. 修改配置文件**

实例：`configs/body_2d_keypoint/rtmpose/mpii/rtmpose-m_8xb64-210e_mpii-256x256.py`
    
```python
dataset_type = 'MpiiDataset'
data_root = '/media/user/T7/MPII/' # 移动硬盘
data_mode = 'topdown'
```

### **4. 使用`browse_dataset`工具浏览数据集**

```shell
python tools/misc/browe_dataset.py <实例配置文件.py> --mode original
# --mode original: 显示原始图片
# --mode transform： 显示增强变换后的图片
```

参考：https://mmpose.readthedocs.io/zh-cn/latest/user_guides/prepare_datasets.html


## 训练和测试

### **1. 训练**

```shell
# 单卡训练
python tools/train.py <实例配置文件.py> [optional arguments]

# CPU
CUDA_VISIBLE_DEVICES=-1 python tools/train.py ...

# 多卡训练
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [PY_ARGS]
```

可选参数参考：https://mmpose.readthedocs.io/zh-cn/latest/user_guides/train_and_test.html

    --amp # 混合精度训练可以减少训练时间和存储需求，而不改变模型或降低模型训练精度，从而支持更大的 batch size、更大的模型和更大的输入尺寸。


### **2. 恢复训练（Resume）**

需要恢复的状态包括：模型权重、优化器状态和优化器参数调整策略的状态

- 自动恢复
用户可以在训练命令的末尾添加 `--resume` 来恢复训练。程序会自动从 work_dirs 中加载最新的权重文件来恢复训练。
如果 work_dirs 中有最新的 checkpoint（例如在之前的训练中中断了训练），则会从 checkpoint 处恢复训练。
```shell
python tools/train.py <实例配置文件>.py --resume
```

- 指定checkpoint恢复
你可以在 配置文件中的 `load_from` 中指定 checkpoint 的路径，并设置`resume = True`，MMPose 会自动读取 checkpoint 并从中恢复训练。
需要注意的是，如果只设置了 `load_from` 而没有设置 `resume=True`，那么只会加载 checkpoint 中的权重，而不会从之前的状态继续训练。
```python
python tools/train.py <cfg.py>  --resume <ckpt.pth>
```

### **3. 冻结参数**

你可以通过在 `paramwise_cfg` 中设置 `custom_keys` 来为模型中的任何模块设置不同的超参数。这样可以让你控制模型特定部分的学习率和衰减系数。

```python
# 在配置文件中设置
# 原始配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# 改为（例如，冻结 backbone.layer0 和 backbone.layer1）
optim_wrapper = dict(
    optimizer=dict(...),
    paramwise_cfg=dict(
        custom_keys={
            'backbone.layer0': dict(lr_mult=0, decay_mult=0),
            'backbone.layer0': dict(lr_mult=0, decay_mult=0),
        }))
```

### **4. 训练结果（生成的文件）**

- 命令行：


    02/05 21:49:37 - INFO - Epoch(train)  [92][100/348]  base_lr: 4.0e-03 lr: 4.000000e-03  eta: 2:58:00  time: 0.25  data_time: 0.057 memory: 5724  loss: 0.533  loss_kpt: 0.533  acc_pose: 0.508

- work_dirs:


    yyyymmdd_HHMMSS
      ├── vis_data
        │   ├── yyyymmdd_HHMMSS.json
        │   ├── config.py
        │   ├── scalars.json
      ├──yyyymmdd_HHMMSS.log
    ├──best_PCK_epoch_92.pth # 最佳模型 metric = PCK, epoch = 92
    ├──latest.pth # 最新模型
    ├──<cfg.py>


### **5. 可视化**

配置文件中设置visualizer
```python
visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
])
```
```shell
tensorboard --logdir work_dirs/xxx/vis_data 
# 暂时还未测试成功
```

### **6. 测试**

```shell
python tools/tests.py <cfg.py> <ckpt.pth> [args]
```
参数参考：https://mmpose.readthedocs.io/zh-cn/latest/user_guides/train_and_test.html

```shell
--work-dir <work_dir> # 指定输出目录
--out <output_file> # 指定输出文件
--cfg-options <OPTIONS># 覆盖部分配置参数
--show # 显示测试结果
--interval <N> # 每隔多少张图片可视化一次
--wait-time <TIME> # 可视化持续时间
```

使用多数据集评估

```python
# 配置文件
# 设置验证数据集
coco_val = dict(type='CocoDataset', ...)
aic_val = dict(type='AicDataset', ...)

val_dataset = dict(
        type='CombinedDataset',
        datasets=[coco_val, aic_val],
        pipeline=val_pipeline,
        ...)

# 配置评估器
val_evaluator = dict(
    type='MultiDatasetEvaluator',
    metrics=[  # 为每个数据集配置度量
        dict(type='CocoMetric',
             ann_file='data/coco/annotations/person_keypoints_val2017.json'),
        dict(type='CocoMetric',
            ann_file='data/aic/annotations/aic_val.json',
            use_area=False,
            prefix='aic')
    ],
    # 数据集个数和顺序与度量必须匹配
    datasets=[coco_val, aic_val],
    )
```
关键点转换详见：https://mmpose.readthedocs.io/zh_CN/dev-1.x/user_guides/mixed_datasets.html#aic-coco



### **7. 输出文件 / 指标复现 / 结果比较**

```shell
# 命令行输出 在work_dirs/yyyymmdd_HHMMSS/yyyymmdd_HHMMSS.json 中保存
 Epoch(test) [93/93]    Head PCK: 95.975443  Shoulder PCK: 94.752038  Elbow PCK: 86.364617  Wrist PCK: 79.305556  Hip PCK: 87.709866  Knee PCK: 81.723960  Ankle PCK: 75.341774  PCK: 86.565183  PCK@0.1: 26.677075  data_time: 0.015544  time: 0.080053
```

```shell
官方指标：
{"Head PCK": 96.92412850307586, "Shoulder PCK": 96.35806671204901, "Elbow PCK": 90.5209635560393, "Wrist PCK": 85.3914664501284, "Hip PCK": 89.62707156720231, "Knee PCK": 86.81040481141916, "Ankle PCK": 83.01753443482589, "PCK": 90.2701715954728, "PCK@0.1": 33.90184120695922, "step": 210}
官方权重文件结果：
Epoch(test) [93/93]    Head PCK: 97.135061  Shoulder PCK: 96.569293  Elbow PCK: 90.864083  Wrist PCK: 86.482026  Hip PCK: 90.219894  Knee PCK: 87.487638  Ankle PCK: 83.325446  PCK: 90.749415  PCK@0.1: 34.806141  data_time: 0.016428  time: 0.077882
```
比较发现官方结果要比自己训练的结果好，原因可能是官方使用了多个数据集进行训练，而我们只使用了MPII数据集。








