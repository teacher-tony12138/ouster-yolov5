import cv2
import numpy as np
from contextlib import closing
from ouster import client
from ouster import pcap
from tqdm import tqdm
import matplotlib.pyplot as plt

metadata_path = './data/lidar/OS2-128_Rev-06.json'
pcap_path = './data/lidar/OS2-128_Rev-06.pcap'
img_path = './output'

with open(metadata_path, 'r') as f:
    metadata = client.SensorInfo(f.read())

source = pcap.Pcap(pcap_path, metadata)

scaler = {
    client.ChanField.RANGE: 255,
    client.ChanField.REFLECTIVITY: 255,
    client.ChanField.SIGNAL: 1024,
    client.ChanField.NEAR_IR: 5120
}


def destagger_img(field):
    want_field = scan.field(field)
    want_val = client.destagger(metadata, want_field)
    scale_num = scaler.get(field, 255)
    want_val = ((want_val / np.max(want_val)) * scale_num).astype(np.uint8)
    return np.dstack((want_val, want_val, want_val))

counter = 0
with closing(client.Scans(source)) as scans:
    for scan in tqdm(scans):
        counter += 1
        range_img = destagger_img(client.ChanField.RANGE)
        ref_img = destagger_img(client.ChanField.REFLECTIVITY)
        ir_img = destagger_img(client.ChanField.NEAR_IR)
        singal_img = destagger_img(client.ChanField.SIGNAL)
        
        cv2.imwrite(f'./output/range/{counter}.jpg', range_img)
        cv2.imwrite(f'./output/ref/{counter}.jpg', ref_img)
        cv2.imwrite(f'./output/ir/{counter}.jpg', ir_img)
        cv2.imwrite(f'./output/singal/{counter}.jpg', singal_img)

# cv2.imshow('image', singal_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

        