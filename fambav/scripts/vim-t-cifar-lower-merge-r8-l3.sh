#!/bin/bash

#SBATCH --job-name=vim-t-cifar-lower-merge-r8-l3          # 作业名称
#SBATCH --account=PAS2490		    # Project ID
#SBATCH --output=/users/PAS2490/marcusshen/Vim/output_logs_vim/vim-t-cifar-lower-merge-r8-l3.log         # 输出日志文件
#SBATCH --error=/users/PAS2490/marcusshen/Vim/output_logs_vim/vim-t-cifar-lower-merge-r8-l3_error.log           # 错误日志文件
#SBATCH --nodes=1                   # 节点数
#SBATCH --ntasks-per-node=1         # 每个节点的任务数
#SBATCH --cpus-per-task=4           # 每个任务使用的 CPU 核心数
#SBATCH --gpus-per-node=4	        # GPU per node
#SBATCH --mem=80G                   # 内存限制
#SBATCH --time=6:00:00             # 作业运行时间限制

# 运行命令或脚本 wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh
source $HOME/miniconda3/bin/activate /users/PAS2490/marcusshen/miniconda3/envs/vim
# module load cuda 
export CUDA_VISIBLE_DEVICES=0,1,2,3

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env ../main.py --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --batch-size 128 --drop-path 0.0 --weight-decay 0.1 --num_workers 25 --data-set CIFAR --data-path ./datasets/cifar-100-python --output_dir ../output/merge_cifar_lower_r8l3_vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --no_amp --fusion-strategy lower --fusion-layer 3 --fusion-token 8