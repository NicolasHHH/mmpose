from mmpose.apis import MMPoseInferencer
# https://mmpose.readthedocs.io/en/dev-1.x/user_guides/inference.html#inferencer-a-unified-inference-interface

# img_path = 'tests/data/coco/000000000785.jpg'   # replace this with your own image path
#
# # instantiate the inferencer using the model alias
# inferencer = MMPoseInferencer('human')
#
# # The MMPoseInferencer API employs a lazy inference approach,
# # creating a prediction generator when given input
# result_generator = inferencer(img_path, show=True)
# result = next(result_generator)
#
#
# # Custom Pose Estimation Models
#
# # build the inferencer with model alias
# inferencer = MMPoseInferencer('human')
#
# # build the inferencer with model config name
# inferencer = MMPoseInferencer('td-hm_hrnet-w32_8xb64-210e_coco-256x192')
#
# # build the inferencer with model config path and checkpoint path/URL
# inferencer = MMPoseInferencer(
#     pose2d='configs/body_2d_keypoint/topdown_heatmap/coco/' \
#            'td-hm_hrnet-w32_8xb64-210e_coco-256x192.py',
#     pose2d_weights='https://download.openmmlab.com/mmpose/top_down/' \
#                    'hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
# )
# # Custom Inferencer for 3D Pose Estimation Models
# # build the inferencer with 3d model alias
# inferencer = MMPoseInferencer(pose3d="human3d")
# # build the inferencer with 3d model config name
# inferencer = MMPoseInferencer(pose3d="motionbert_dstformer-ft-243frm_8xb32-120e_h36m")


# result_generator = inferencer(img_path, pred_out_dir='predictions')
# result = next(result_generator)
#
# result_generator = inferencer(img_path, out_dir='output')
# result = next(result_generator)

def main():

    # build the inferencer with 3d model config path and checkpoint path/URL
    inferencer = MMPoseInferencer(
        pose3d='configs/body_3d_keypoint/motionbert/h36m/' \
               'motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py',
        pose3d_weights='https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/' \
                       'pose_lift/h36m/motionbert_ft_h36m-d80af323_20230531.pth'
    )

    result_generator = inferencer("/home/user/PycharmProjects/mmpose/tests/data/panoptic_body3d/160906_band1", show=True)
    result = next(result_generator)

if __name__ == '__main__':
    main()