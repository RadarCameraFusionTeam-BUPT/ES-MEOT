import os, sys
from utils.Space2Plane import Space2Plane
import cv2
import numpy as np
from utils.TPMB_RCfusion import rotation_vectors_to_matrices
from utils.parameters import *
from tqdm import tqdm

ROOT_DIR = os.path.abspath("./")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

config_path = '/mnt/datasets/radar_camera_bridge/2/config'
video_path = '/mnt/datasets/radar_camera_bridge/2/output.avi'
track_npy = '/mnt/jiayindeng_work/ES-MEOT-RadarCameraFusion-3d-v2/output/result.npy'

# config_path = '/data/jiayindeng_work/ES-MEOT-RadarCameraFusion/simulation/config'
# video_path = '/data/jiayindeng_work/ES-MEOT-RadarCameraFusion/simulation/output.mp4'
# track_npy = '/data/jiayindeng_work/ES-MEOT-RadarCameraFusion/output/result.npy'

s2p = Space2Plane(config=config_path)

fx, fy = s2p.cameraMatrix[0, 0], s2p.cameraMatrix[1, 1]
u0, v0 = s2p.cameraMatrix[0, 2], s2p.cameraMatrix[1, 2]

cap = cv2.VideoCapture(video_path)

data = np.load(track_npy, allow_pickle=True)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_name = os.path.join(OUTPUT_DIR, 'result.mp4')
out = cv2.VideoWriter(out_name, fourcc, 20.0, (width, height))

def get_text_color(pixel_color, threshold=70):
    # Convert RGB to YUV color space
    yuv_color = cv2.cvtColor(np.uint8([[pixel_color]]), cv2.COLOR_BGR2YUV)[0][0]
    y = yuv_color[0]  # Extracting brightness values

    # Determine text color based on brightness value
    if y > threshold:
        return (0, 0, 0)  # black
    else:
        return (255, 255, 255)  # white


print('Start saving results...')
for now in tqdm(data, total=len(data)):
    _, frame = cap.read()
    if not _:
        break
    pos = now['x_ref'][now['r'] > 0.5, :3]
    theta = now['x_ref'][now['r'] > 0.5, 4:7]
    R = rotation_vectors_to_matrices(theta)
    ext = now['x_ref'][now['r'] > 0.5, 10:13]

    t = now['t'][now['r'] > 0.5]
    t_id = now['t_id'][now['r'] > 0.5]

    # print(now['x_ref'][now['r'] > 0.5, 3])

    corner_point = corner_to_extend[np.newaxis, :, :] * ext[:, np.newaxis, :]
    corner_world = pos[:, np.newaxis, :] + (R @ corner_point.transpose(0, 2, 1)).transpose(0, 2, 1)
    corner_pix = s2p.xcyczc2uv(corner_world.reshape([-1, 3])).reshape([-1, 8, 2])


    u = now['mu'][now['r'] > 0.5, :, 3:6]
    vartheta = pos[:, np.newaxis, :] + (R @ u.transpose(0, 2, 1)).transpose(0, 2, 1)
    proj = s2p.xcyczc2uv(pos)


    u_proj = s2p.xcyczc2uv(vartheta.reshape([-1, 3])).reshape([-1, 24, 2])
    for p, kps, cor_pix, trans, t_, t_id_ in zip(proj, u_proj, corner_pix, pos, t, t_id):
        u, v = p
        # Get the color of the current pixel
        u_c, v_c = min(u, 1280-1), min(v, 720-1)
        pixel_color = np.mean(np.array(frame)[int(v_c)-50:int(v_c)+50, int(u_c)-50:int(u_c)+50, :], axis=(0, 1)).astype(int)

        cv2.circle(frame, (int(u), int(v)), 5, (0, 0, 255), thickness=-1)
        # for kp in kps:
        #     u, v = kp
        #     cv2.circle(frame, (int(u), int(v)), 3, (0, 255, 0), thickness=-1)
        for kp in cor_pix:
            u, v = kp
            cv2.circle(frame, (int(u), int(v)), 3, (0, 255, 255), thickness=-1)
        # cv2.putText(frame, '{:.1f} {:.1f} {:.1f}'.format(trans[0], trans[1], trans[2]), (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Get text color
        text_color = get_text_color(pixel_color)
        
        # Draw text on frames and determine text color based on pixel color
        cv2.putText(frame, '{}:{}'.format(int(t_), int(t_id_)), (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)


    # frame = cv2.resize(frame, (1280, 720))
    out.write(frame)

    # cv2.imshow('img', frame)
    # key = cv2.waitKey(1)
    # if key == 27:
    #     break

print('Save to {}'.format(out_name))
out.release()
cap.release()
cv2.destroyAllWindows()