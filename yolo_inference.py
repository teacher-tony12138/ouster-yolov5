import argparse
import os
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, LoadWebcam, LoadNumpy
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective

from sklearn.cluster import KMeans

from ouster import client
from ouster import pcap
from contextlib import closing
import logging


weights=ROOT / 'best.pt'
imgsz=640  # inference size (pixels)
conf_thres=0.25  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=1000  # maximum detections per image
device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
view_img=True  # show results
save_txt=False  # save results to *.txt
save_conf=False  # save confidences in --save-txt labels
save_crop=False  # save cropped prediction boxes
nosave=False  # do not save images/videos
classes=None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False  # class-agnostic NMS
augment=False  # augmented inference
visualize=False  # visualize features
update=False  # update all models
project=ROOT / 'runs/detect'  # save results to project/name
name='exp'  # save results to project/name
exist_ok=False  # existing project/name ok, do not increment
line_thickness=1  # bounding box thickness (pixels)
hide_labels=False  # hide labels
hide_conf=True  # hide confidences
half=False  # use FP16 half-precision inference
dnn=False  # use OpenCV DNN for ONNX inference
car_distance=True

# Load model
device = select_device(device)
model = DetectMultiBackend(weights, device=device, dnn=dnn)
stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
imgsz = check_img_size(imgsz, s=stride)  # check image size

# Half
half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
if pt:
    model.model.half() if half else model.model.float()

def detect_car(client, scan, pcap_file, metadata):

    ref_field = scan.field(client.ChanField.REFLECTIVITY)
    ref_val = client.destagger(pcap_file.metadata, ref_field)
    #ref_img = (ref_val / np.max(ref_val) * 255).astype(np.uint8)
    ref_img = ref_val.astype(np.uint8)

    range_field = scan.field(client.ChanField.RANGE)
    range_val = client.destagger(pcap_file.metadata, range_field)
    #range_img = (range_val / np.max(range_val) * 255).astype(np.uint8)
    #range_img = range_val

    combined_img = np.dstack((ref_img, ref_img, ref_img))

    xyzlut = client.XYZLut(metadata)
    xyz_destaggered = client.destagger(metadata, xyzlut(scan))

    #run inference
    dataset = LoadNumpy(numpy=combined_img, path="", img_size=imgsz, stride=stride, auto=pt and not jit)

    # if pt and device.type != 'cpu':
    #     model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1

    # Inference
    pred = model(im, augment=augment)
    t3 = time_sync()
    dt[1] += t3 - t2

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    dt[2] += time_sync() - t3

    # Range ROI
    range_roi_list = []

    for i, det in enumerate(pred):  # per image
        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

        p = Path(p)  # to Path
        s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for save_crop
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            poi_list = []
            xyz_list = []
            xyxy_list = []
            range_list = []

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                xyxy_list.append(xyxy)

                if view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    x1 = int(xyxy[0])
                    y1 = int(xyxy[1])
                    x2 = int(xyxy[2])
                    y2 = int(xyxy[3])

                    # range_roi = range_val[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])] #whole box
                    range_roi = range_val[y1: y2, x1: x2]

                    range_roi_list.append([x1, y1, x2, y2])

                    def remove_outliners(target_roi, placeholder=999999):
                        flat = target_roi.reshape(-1)
                        p10 = np.percentile(flat, 10)
                        p90 = np.percentile(flat, 90)
                        target_roi[np.where(target_roi < p10)] = placeholder
                        target_roi[np.where(target_roi > p90)] = placeholder
                        
                        return target_roi
                        
                    range_roi = remove_outliners(range_roi) 
                    range_roi[np.where(range_roi == 0)] = 999999 
                                
                    min_range = np.min(range_roi) 
                    # print(f'min range is {min_range}')
                    # range_list.append(min_range)

                    poi_roi = np.unravel_index(range_roi.argmin(), range_roi.shape) #(y,x) in roi
                    poi_x = poi_roi[1] + x1
                    poi_y = poi_roi[0] + y1
                    poi = (poi_y, poi_x) #(y,x) in global
                    poi_list.append(poi)

                    if car_distance == False:
                        annotator.box_label(xyxy, label, color=colors(c, True))
            
                    xyz_val = xyz_destaggered[poi]
                    xyz_list.append(xyz_val)

            if car_distance == True:

                for idx, xyz in enumerate(xyz_list):
                    cur_x = xyz[0]
                    cur_y = xyz[1]
                    cur_z = xyz[2]

                    import math
                    dist = math.sqrt(cur_x**2 + cur_y**2 + cur_z**2)

                    annotator.display_distance(xyxy_list[idx], poi_list[idx], label, dist, color=colors(c, True))

        # Stream results
        im0 = annotator.result()
        return im0, range_roi_list
