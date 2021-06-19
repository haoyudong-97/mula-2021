## ADE20K
# pair loss
#CUDA_VISIBLE_DEVICES=0 python3 train.py --dataroot ../dataset_cvpr/ade20k --dataset_mode imagetext --model joint --name ade20k_10k --netG joint --batch_size 30 #--lr 0.00001
#CUDA_VISIBLE_DEVICES=0 python3 train.py --dataroot ../dataset_cvpr/ade20k --dataset_mode imagetext --model joint --name ade20k_2k --netG joint --batch_size 30 #--lr 0.00001
#CUDA_VISIBLE_DEVICES=0 python3 train.py --dataroot ../dataset_cvpr/ade20k --dataset_mode imagetext --model joint --name ade20k_3k --netG joint --batch_size 30 #--lr 0.00001
#CUDA_VISIBLE_DEVICES=0 python3 train.py --dataroot ../dataset_cvpr/ade20k --dataset_mode imagetext --model joint --name ade20k_5k --netG joint --batch_size 30 #--lr 0.00001
#CUDA_VISIBLE_DEVICES=1 python3 train.py --dataroot ../dataset_cvpr/ade20k --dataset_mode imagetext --model joint --name ade20k_20k --netG joint --batch_size 30 #--lr 0.00001

#CUDA_VISIBLE_DEVICES=0 python3 train.py --dataroot ../dataset_cvpr/ade20k --dataset_mode imagetext --model joint --name ade20k_5k_large --netG jointlarge --batch_size 50 #--lr 0.00001
#CUDA_VISIBLE_DEVICES=0 python3 train.py --dataroot ../dataset_cvpr/ade20k --dataset_mode imagetext --model joint --name ade20k_k10_large --netG jointlarge --batch_size 50 #--lr 0.002

# nce loss
#CUDA_VISIBLE_DEVICES=0 python3 train.py --dataroot ../dataset_cvpr/ade20k --dataset_mode imagetext --model joint --name ade20k_nce --netG joint --batch_size 30 --n_epochs 2000
#CUDA_VISIBLE_DEVICES=0 python3 train.py --dataroot ../dataset_cvpr/ade20k --dataset_mode imagetext --model joint --name ade20k_nce_full --netG joint --batch_size 30 --n_epochs 2000
#CUDA_VISIBLE_DEVICES=0 python3 train.py --dataroot ../dataset_cvpr/ade20k --dataset_mode imagetext --model joint --name ade20k_nce_largerlr --netG joint --batch_size 50 --n_epochs 2000 --lr 0.01

# supervised
#CUDA_VISIBLE_DEVICES=1 python3 train_supervised.py --dataroot ../dataset_cvpr/ade20k --dataset_mode imagetext --model supervised --name ade20k_supervised --netG supervised --batch_size 50 --n_epochs 2000 --supervised
CUDA_VISIBLE_DEVICES=0 python3 train_supervised.py --dataroot ../dataset_cvpr/ade20k --dataset_mode imagetext --model supervised --name ade20k_supervised_large --netG supervised --batch_size 40 --n_epochs 2000 --supervised

# -----------------
# COCO
#CUDA_VISIBLE_DEVICES=0 python3 train.py --dataroot ../dataset_cvpr/coco2017 --dataset_mode imagetext --model joint --name coco2017 --netG joint --batch_size 200 #--lr 0.00001
#CUDA_VISIBLE_DEVICES=1 python3 train.py --dataroot ../dataset_cvpr/coco2017 --dataset_mode imagetext --model joint --name coco2017_k100 --netG joint --batch_size 200 #--lr 0.002
#CUDA_VISIBLE_DEVICES=0 python3 train.py --dataroot ../dataset_cvpr/coco2017 --dataset_mode imagetext --model joint --name coco2017_k50 --netG joint --batch_size 200 #--lr 0.00001
#CUDA_VISIBLE_DEVICES=0 python3 train.py --dataroot ../dataset_cvpr/coco2017 --dataset_mode imagetext --model joint --name coco2017_k30 --netG joint --batch_size 200 #--lr 0.002
#CUDA_VISIBLE_DEVICES=1 python3 train.py --dataroot ../dataset_cvpr/coco2017 --dataset_mode imagetext --model joint --name coco2017_k10_sgd --netG joint --batch_size 200 --lr 0.002

# ----------------
# Openimage
#CUDA_VISIBLE_DEVICES=1 python3 train.py --dataroot ../dataset_cvpr/openimage --dataset_mode imagetext --model joint --name openimage_k5 --netG joint --batch_size 240 --lr 0.002 --beta1 0.9
#CUDA_VISIBLE_DEVICES=0 python3 train.py --dataroot ../dataset_cvpr/openimage --dataset_mode imagetext --model joint --name openimage_k5 --netG joint --batch_size 200 #--lr 0.002
#CUDA_VISIBLE_DEVICES=0 python3 train.py --dataroot ../dataset_cvpr/openimage --dataset_mode imagetext --model joint --name openimage_k5_sgd --netG joint --batch_size 200 --lr 0.002
#CUDA_VISIBLE_DEVICES=0 python3 train.py --dataroot ../dataset_cvpr/openimage --dataset_mode imagetext --model joint --name openimage_k10_sgd --netG joint --batch_size 200 --lr 0.002
#CUDA_VISIBLE_DEVICES=1 python3 train.py --dataroot ../dataset_cvpr/openimage --dataset_mode imagetext --model joint --name openimage_k10 --netG joint --batch_size 200 #--lr 0.002
#CUDA_VISIBLE_DEVICES=0 python3 train.py --dataroot ../dataset_cvpr/openimage --dataset_mode imagetext --model joint --name openimage_k15 --netG joint --batch_size 200 #--lr 0.002
#CUDA_VISIBLE_DEVICES=1 python3 train.py --dataroot ../dataset_cvpr/openimage --dataset_mode imagetext --model joint --name openimage_k20 --netG joint --batch_size 200 #--lr 0.002
#CUDA_VISIBLE_DEVICES=0 python3 train.py --dataroot ../dataset_cvpr/openimage --dataset_mode imagetext --model joint --name openimage_k50 --netG joint --batch_size 200 #--lr 0.002

#CUDA_VISIBLE_DEVICES=1 python3 train.py --dataroot ../dataset_cvpr/openimage --dataset_mode imagetext --model joint --name openimage_k5-large --netG jointlarge --batch_size 50 #--lr 0.002
#CUDA_VISIBLE_DEVICES=1 python3 train.py --dataroot ../dataset_cvpr/openimage --dataset_mode imagetext --model joint --name openimage_k10_large --netG jointlarge --batch_size 50 #--lr 0.002

#CUDA_VISIBLE_DEVICES=1 python3 train.py --dataroot ../dataset_cvpr/openimage --dataset_mode imagetext --model joint --name openimage_nce --netG joint --batch_size 250 --lr 0.0001 --continue_train
# supervised
#CUDA_VISIBLE_DEVICES=1 python3 train_supervised.py --dataroot ../dataset_cvpr/openimage --dataset_mode imagetext --model supervised --name openimage_supervised --netG supervised --batch_size 50 --n_epochs 2000 --supervised
