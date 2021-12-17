
###
 # @Author: Xiang Pan
 # @Date: 2021-12-14 18:46:35
 # @LastEditTime: 2021-12-14 19:36:00
 # @LastEditors: Xiang Pan
 # @Description: 
 # @FilePath: /project/scripts/low_kitti.sh
 # @email: xiangpan@nyu.edu
### 
python main.py --gpu 0 --task_name=KITTI --load_checkpoint_path ./cached_models/unity-streetview-low-res_mIOU=0.77.ckpt --log_name=low_kitti