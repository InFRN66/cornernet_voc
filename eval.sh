
#! /bin/bash

# python test.py CornerNet --testiter 500000 --split testing
python eval.py \
--cfg_file originalNet \
--dataset /mnt/hdd01/img_data/VOCdevkit/VOC2007 \
--eval_dir ./eval \
--testiter 5000 \
--mAP_iou_threshold 0.5