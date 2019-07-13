#!/usr/bin/env python
import os
import json
import torch
import pprint
import argparse
import importlib
import numpy as np
import pathlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from config import system_configs_voc
from nnet.py_factory import NetworkFactory
from db.datasets import datasets

from utils.eval_utils.voc_dataset import VOCDataset
from utils.eval_utils import measurements, box_utils

# torch.backends.cudnn.benchmark = False
torch.backends.cudnn.benchmark = True

def str2bool(s):
    return s.lower() in ('true', '1')

def parse_args():
    parser = argparse.ArgumentParser(description="Test CornerNet")
    parser.add_argument("--cfg_file", help="config file", type=str)
    parser.add_argument("--label_file", type=str, default="../MIRU2019/models/voc-model-labels.txt", help="The label file path.")
    parser.add_argument("--dataset", type=str, help="The root directory of the VOC dataset or Open Images dataset.")
    parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store evaluation results.")
    parser.add_argument("--mAP_iou_threshold", type=float, default=0.5, help="The threshold for mAP measurement.")
    parser.add_argument("--use_2007_metric", type=str2bool, default=True)
    parser.add_argument("--testiter", dest="testiter", help="test at iteration i",
                        default=None, type=int)
    args = parser.parse_args()
    return args

def make_dirs(directories):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def get_image_files(image_dir):
    image_file_list = sorted(list(image_dir.glob("*.jpg")))
    return image_file_list


def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i) # len(annotation) = 3 , (coords[num_target, 4], labels[num_target], is_difficult[num_target])
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index]={}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id]) # [1, 4]
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
    return true_case_stat, all_gt_boxes, all_difficult_cases # each class, each image(id)の情報


def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):

    """
    num_true_cases : 実際にそのクラスに属するtarget総数
    """
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f: # そのクラス分のprediction box 全部読み込み
            t = line.rstrip().split(" ") # t: [image_id, scores, boxes]
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores) # -付きなのでlarger is firstでindex // 全ボックス内でconfidenceの高い順にsort
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids): # image_ids : このクラスと判断されたimageのid / gt_boxes : 実際にこのクラスだったボックス
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1 # このクラスの物体box，と予測して実際は違った
                continue

            gt_box = gt_boxes[image_id] # 多分複数ボックス（そのimage_id中のこのクラスの正解ボックス）
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold: # 少なくともどれかのgtboxに対してIoU>しきい値であったpredictionが存在
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched: # これまで，同じimage_id中で今見てるボックス（max_arg）が検出されていない場合(not in match)
                        true_positive[i] = 1 # max_argのボックス = true positive
                        matched.add((image_id, max_arg))
                    else: # 同じgt_boxが既に検出され，true positiveとして数えられている:　同じ物体を重複してpredicしている場合など．
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return measurements.compute_average_precision(precision, recall)


def eval():
    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)
    class_names = [name.strip() for name in open(args.label_file).readlines()]

    print("loading parameters at iteration: {}".format(args.testiter))
    print("building neural network...")
    nnet = NetworkFactory(db=None, dataset="voc")
    print("loading parameters...")
    nnet.load_params(args.testiter)
    nnet.cuda()
    nnet.eval_mode()
    predictor = importlib.import_module("test.voc").kp_detection_oneImage
    
    dataset = VOCDataset(args.dataset, is_test=True)
    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)

    results = []
    for i in range(len(dataset)):
        print("process image", i)
        image = dataset.get_image(i) # original [h,w,c]
        boxes, labels, probs = predictor(nnet, image)  # [num_box, 4], [num_box], [num_box] // nms後のもの. prob_thrshold=..で無視するボックス設定
        
        boxes = torch.from_numpy(boxes)
        labels = torch.from_numpy(labels)
        probs = torch.from_numpy(probs)
        # print(probs)
        # print(labels)
        # exit()
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i # [num_bb, 1], value=i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes + 1.0  # matlab's indexes start from 1
        ], dim=1)) # results[i] = [num_box, 7]in columns, [index, label, prob, boxes+1]
    results = torch.cat(results)
    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue  # ignore background
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :] # class indexに該当する箇所だけ選択
            for i in range(sub.size(0)): # num_boxes in  this class
                prob_box = sub[i, 2:].numpy()
                image_id = dataset.ids[int(sub[i, 0])]
                print(
                    image_id + " " + " ".join([str(v) for v in prob_box]),
                    file=f
                )

    f = open(os.path.join(args.eval_dir, 'all.txt'), 'w')
    aps = []
    print("\n\nAverage Precision Per-class:")
    f.write("Average Precision Per-class:\n\n")
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        ap = compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            all_difficult_cases[class_index],
            prediction_path,
            args.mAP_iou_threshold,
            args.use_2007_metric
        )
        aps.append(ap)
        print(f"{class_name}: {ap}")
        f.write(f"{class_name}: {ap}\n")

    print(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")
    f.write(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")


if __name__ == "__main__":
    args = parse_args()

    
    cfg_file = os.path.join(system_configs_voc.config_dir, args.cfg_file + ".json")
    print("cfg_file: {}".format(cfg_file)) # -- ./config/originalNet.json

    with open(cfg_file, "r") as f:
        configs = json.load(f)

    configs["system"]["snapshot_name"] = args.cfg_file # -- originalNet
    system_configs_voc.update_config(configs["system"])

    print("loading all datasets...")

    eval()