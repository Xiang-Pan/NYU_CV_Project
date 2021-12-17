###
 # @Author: Xiang Pan
 # @Date: 2021-11-12 09:52:35
 # @LastEditTime: 2021-12-14 23:07:43
 # @LastEditors: Xiang Pan
 # @Description: 
 # @FilePath: /project/scripts/all.sh
 # @email: xiangpan@nyu.edu
### 
python main.py --gpu 0 --task_name=KITTI --load_checkpoint_path=./cached_models/unity-cameraview-high-res_mIOU=0.38.ckpt --log_name=cameraview_high_kitti
python main.py --gpu 0 --task_name=KITTI --load_checkpoint_path=./cached_models/unity-cameraview-low-res-mIOU=0.40.ckpt --log_name=cameraview_low_kitti