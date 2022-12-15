# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on pcap

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --meta-data path/*.json --source path/*.pcap  # directory
                                                             
"""

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

@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=1,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=True,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        car_distance=False,
        metadata_path=ROOT / 'example.json'
        ):
    source = str(source)
    is_pcap = source.endswith('.pcap')

    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Dataloader
    if is_pcap:
        print('pcap file')
        
        metadata_path = str(metadata_path)

        with open(metadata_path, 'r') as f:
            metadata = client.SensorInfo(f.read())

        fps = int(str(metadata.mode)[-2:])
        print('fps: ', fps)
        width = int(str(metadata.mode)[:4])
        print('width: ', width)
        height = int(str(metadata.prod_line)[5:])
        print('height: ', height)

        pcap_file = pcap.Pcap(source, metadata)
        
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        with closing(client.Scans(pcap_file)) as scans:

            save_path = str(save_dir/"results.mp4")  # im.jpg
            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            bs = 1 # batch_size

            for scan in scans:
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

                if pt and device.type != 'cpu':
                    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
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
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=augment, visualize=visualize)
                t3 = time_sync()
                dt[1] += t3 - t2

                # NMS
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                dt[2] += time_sync() - t3

                # Second-stage classifier (optional)
                # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

                # Process predictions

                for i, det in enumerate(pred):  # per image
                    seen += 1
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f'{i}: '
                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    #save_path = str(save_dir / p.name)  # im.jpg
                    #txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                    txt_path = str(save_dir / 'labels' / p.stem) 
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

                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')

                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                x1 = int(xyxy[0])
                                y1 = int(xyxy[1])
                                x2 = int(xyxy[2])
                                y2 = int(xyxy[3])

                                # range_roi = range_val[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])] #whole box
                                range_roi = range_val[y1: y2, x1: x2]

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
                    
                                if save_crop:
                                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                        if car_distance == True:

                            for idx, xyz in enumerate(xyz_list):
                                cur_x = xyz[0]
                                cur_y = xyz[1]
                                cur_z = xyz[2]

                                import math
                                dist = math.sqrt(cur_x**2 + cur_y**2 + cur_z**2)

                                annotator.display_distance(xyxy_list[idx], poi_list[idx], label, dist, color=colors(c, True))
                    
                    # Print time (inference-only)
                    LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

                    # Stream results
                    im0 = annotator.result()
                    if view_img:
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(100)  # 1 millisecond


                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                    vid_writer.write(im0)    

            vid_writer.release()

            # Print results
            t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
            if save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
            if update:
                strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=1, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--car-distance', action='store_true', help='calculate distance between lidar and cars')
    parser.add_argument('--metadata-path', type=str, default=ROOT / 'example.json', help='metadata path')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt    


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
