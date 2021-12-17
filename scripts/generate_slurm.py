'''
Author: Xiang Pan
Date: 2021-12-08 03:10:47
LastEditTime: 2021-12-14 19:45:55
LastEditors: Xiang Pan
Description: 
FilePath: /project/scripts/generate_slurm_by_dir.py
@email: xiangpan@nyu.edu
'''
import os

dirs = os.listdir("./scripts")
import os
templace_name = "./scripts/template.sh"

src_file = open("./scripts/high_kitti.sh", "r")


for line in src_file:
    if "python" in line:
        # print(line)
        prop = line.split(" ")[3]
        split = line.split(" ")[5]
        task = line.split(" ")[7]
        print(task)
        method = line.split(" ")[-1].replace("&", "").strip()
        print(method)
        name = prop + "_" + split + "_" + task + "_" + method
        tgt_name = "./slurm_scripts/" + name + ".sh"
        # print(prop, split)
        os.system("cp "+templace_name + " " + tgt_name)
        line = line.replace("&\n", "")
        tgt_file = open(tgt_name, "a+")
        tgt_file.write(line)
        
        # os.system("echo {} >> {}".format(line, tgt_name))
        # break

# for fpathe,dirs,fs in os.walk('./scripts'):
    # for f in fs:
    #     src_name = os.path.join(fpathe,f)
    #     tgt_name = src_name.replace("./scripts/","./slurm_scripts/")
    #     dir_name = "/".join(tgt_name.split("/")[0:-1])
    #     if not os.path.exists(dir_name):
    #         os.makedirs(dir_name)
    #     if os.path.exists(tgt_name):
    #         os.system("rm {}".format(tgt_name))
    #     os.system("cp "+templace_name + " " + tgt_name)
    #     os.system("echo {} >> {}".format(src_name,tgt_name))
    #     # print(dir_name)