# ade20k
#CUDA_VISIBLE_DEVICES=0 python3 evaluate.py --dataroot ../dataset_cvpr/ade20k --dataset_mode imagetext --model joint --name ade20k --netG joint --batch_size 30 --continue_train
#CUDA_VISIBLE_DEVICES=0 python3 evaluate.py --dataroot ../dataset_cvpr/ade20k --dataset_mode imagetext --model joint --name ade20k_2k --netG joint --batch_size 30 --continue_train
#CUDA_VISIBLE_DEVICES=1 python3 evaluate.py --dataroot ../dataset_cvpr/ade20k --dataset_mode imagetext --model joint --name ade20k_3k --netG joint --batch_size 30 --continue_train
#CUDA_VISIBLE_DEVICES=0 python3 evaluate.py --dataroot ../dataset_cvpr/ade20k --dataset_mode imagetext --model joint --name ade20k_5k --netG joint --batch_size 30 --continue_train
#CUDA_VISIBLE_DEVICES=1 python3 evaluate.py --dataroot ../dataset_cvpr/ade20k --dataset_mode imagetext --model joint --name ade20k_10k --netG joint --batch_size 30 --continue_train

#CUDA_VISIBLE_DEVICES=0 python3 evaluate.py --dataroot ../dataset_cvpr/ade20k --dataset_mode imagetext --model joint --name ade20k_nce --netG joint --batch_size 30 --continue_train
#CUDA_VISIBLE_DEVICES=0 python3 evaluate.py --dataroot ../dataset_cvpr/ade20k --dataset_mode imagetext --model joint --name ade20k_nce_full --netG joint --batch_size 30 --continue_train
#CUDA_VISIBLE_DEVICES=1 python3 evaluate.py --dataroot ../dataset_cvpr/ade20k --dataset_mode imagetext --model joint --name ade20k_nce_largerlr --netG joint --batch_size 100 --continue_train

CUDA_VISIBLE_DEVICES=0 python3 evaluate.py --dataroot ../dataset_cvpr/ade20k --dataset_mode imagetext --model joint --name ade20k_k10_large --netG jointlarge --batch_size 50 --continue_train

#--------
# OpenImage
#CUDA_VISIBLE_DEVICES=1 python3 evaluate.py --dataroot ../dataset_cvpr/openimage --dataset_mode imagetext --model joint --name openimage_k20 --netG joint --batch_size 200 #--lr 0.002
#CUDA_VISIBLE_DEVICES=0 python3 evaluate.py --dataroot ../dataset_cvpr/openimage --dataset_mode imagetext --model joint --name openimage_k50 --netG joint --batch_size 200 #--lr 0.002
