
###
 # @Author: Xiang Pan
 # @Date: 2021-12-14 18:46:35
 # @LastEditTime: 2021-12-14 19:36:19
 # @LastEditors: Xiang Pan
 # @Description: 
 # @FilePath: /project/scripts/high_kitti.sh
 # @email: xiangpan@nyu.edu
### 
python main.py --gpu 0 --task_name=KITTI --load_checkpoint_path=./cached_models/unity-streetview-high-res_mIOU=0.73.ckpt --log_name=high_kitti