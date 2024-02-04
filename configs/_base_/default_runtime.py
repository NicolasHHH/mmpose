# 通用配置
default_scope = 'mmpose'

# hooks
default_hooks = dict(
    # 迭代计时器，包括数据耗时、模型耗时
    timer=dict(type='IterTimerHook'),

    # 日志记录器，默认50iters打印一次
    logger=dict(type='LoggerHook', interval=50),

    # 调度学习率更新
    param_scheduler=dict(type='ParamSchedulerHook'),

    # 设置ckpt保存间隔，最优ckpt判断指标
    # 例如： save_best_ckpt = “coco/AP”
    # mmpose/evaluation/metrics
    checkpoint=dict(type='CheckpointHook', interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # cudnn benchmark = False 用于加速训练，但是会增加显存
    visualization=dict(type='PoseVisualizationHook', enable=False),
    badcase=dict(
        type='BadCaseAnalysisHook',
        enable=False,
        out_dir='badcase',
        metric_type='loss',
        badcase_thr=5))

# custom hooks
custom_hooks = [
    # Synchronize model buffers such as running_mean and running_var in BN
    # at the end of each epoch
    dict(type='SyncBuffersHook')
]

# multi-processing backend
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# visualizer
vis_backends = [
    dict(type='LocalVisBackend'),
    # dict(type='TensorboardVisBackend'),
    # dict(type='WandbVisBackend'),
]
visualizer = dict(
    type='PoseLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# logger
log_processor = dict(
    type='LogProcessor', window_size=50, by_epoch=True, num_digits=6)
# 日志记录等级，INFO 代表记录训练日志，WARNING 代表只记录警告信息，ERROR 代表只记录错误信息
log_level = 'INFO'
load_from = None
resume = False

# file I/O backend
backend_args = dict(backend='local')

# training/validation/testing progress
train_cfg = dict(by_epoch=True)
val_cfg = dict()
test_cfg = dict()
