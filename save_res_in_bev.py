import cv2
import os
import numpy as np
from utils.TPMB_RCfusion import rotation_vectors_to_matrices
from utils.parameters import *
from tqdm import tqdm

ROOT_DIR = os.path.abspath("./")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

track_npy = './results_all_methods/ES-MEOT/result.npy'
data = np.load(track_npy, allow_pickle=True)

sensor_to_world_R = np.array([[ 0.99959675, -0.02111941,  0.01898153],
                               [-0.02229995, -0.17001168,  0.9851897 ],
                               [-0.01757955, -0.98521571, -0.17041409]])
sensor_to_world_T = np.array([0., 0., 6.87052665])

out_name = 'BEV.mp4'
video_path = os.path.join(OUTPUT_DIR, out_name)

# Define video output parameters
fps = 20
frame_width = 400
frame_height = 1700
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height))

r_thres = 0.0

print('Start saving results...')
for frame_idx, now in tqdm(enumerate(data), total=len(data)):
    frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255  # 白色背景
    if len(now['r']) == 0:
        continue

    pos = now['x_ref'][now['r'] > r_thres, :3]
    theta = now['x_ref'][now['r'] > r_thres, 4:7]
    R = rotation_vectors_to_matrices(theta)
    ext = now['x_ref'][now['r'] > r_thres, 10:13]

    t = now['t'][now['r'] > r_thres]
    t_id = now['t_id'][now['r'] > r_thres]

    corner_point = corner_to_extend[np.newaxis, :, :] * ext[:, np.newaxis, :]
    corner_sensor = pos[:, np.newaxis, :] + (R @ corner_point.transpose(0, 2, 1)).transpose(0, 2, 1)
    corner_world = sensor_to_world_T[np.newaxis, np.newaxis, :] + (sensor_to_world_R @ corner_sensor.transpose(0, 2, 1)).transpose(0, 2, 1)

    # Draw Corner_world onto the frame
    corner_world_to_pix = (corner_world[:, :4, :2] * 10).astype(int)
    corner_world_to_pix[:, :, 0] += int(frame_width / 2)
    corner_world_to_pix[:, :, 1] = frame_height - corner_world_to_pix[:, :, 1]

    for j, obj in enumerate(corner_world_to_pix):
        mid = np.max(obj, axis=0).astype(int)
        cv2.putText(frame, '{}:{}'.format(int(t[j]), int(t_id[j])), mid, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        for i in range(3):
            try:
                cv2.line(frame, obj[i], obj[(i+1)%4], color=(0, 0, 255), thickness=5)
            except:
                pass

    # Write frame into video
    out.write(frame)

# Release video writer
out.release()

print('Save to {}'.format(video_path))
