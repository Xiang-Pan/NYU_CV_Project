###
 # @Author: Xiang Pan
 # @Date: 2021-08-23 03:16:28
 # @LastEditTime: 2021-08-23 03:26:55
 # @LastEditors: Xiang Pan
 # @Description: 
 # @FilePath: /CommonsenseQA/slrum_scripts/sbatch_all.sh
 # xiangpan@nyu.edu
### 
#！/bin/bash
dir_name=$1
for file in $(ls $dir_name)
do 
    if [ "${file##*.}" = "sh" ]; then
        # echo $file
        echo $dir_name/$file
        sbatch $dir_name/$file
        # mv ${file} ${file%.*}.c
    fi
done