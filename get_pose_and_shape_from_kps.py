import os
import numpy as np
import cv2
import argparse
import configparser
import json
from utils import car_model
import multiprocessing as mp
from tqdm import tqdm

ROOT_DIR = os.path.abspath("./")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

# Create the OUTPUT_DIR folder
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

def solve(item):
    rvecs = []
    sizes = []
    skeletons = []

    for kps in item['keypoints']:
        kps = np.array(kps)
        if len(kps) < 8:
            rvecs.append(np.array([0, 0, 0]))
            sizes.append(np.array([1, 1, 1]))
            skeletons.append(np.array(list(car_model.world_car.values())[0]).astype(np.float64))
            continue

        valid_id = kps[:, 0].astype(int)

        min_error = float('inf')
        min_car = None
        min_tvec = None
        min_rvec = None
        min_size = None

        for car in car_model.world_car_ext.keys():
            car_points = np.array(
                car_model.world_car_ext[car]).astype(np.float64)
            car_points[:, 1] *= -1

            try:
                _, rvec, tvec, inliers = cv2.solvePnPRansac(car_points[valid_id], kps[:, 1:].astype(np.float64), K, distCoeffs, flags=cv2.SOLVEPNP_EPNP)
            except:
                continue

            tmp_img = cv2.projectPoints(
                car_points[valid_id], rvec, tvec, K, distCoeffs)[0][:, 0, :]

            error = np.mean(np.linalg.norm(tmp_img - kps[:, 1:], axis=1))
            if error < min_error:
                min_error = error
                min_car = car
                min_tvec = tvec
                min_rvec = rvec
                min_size = np.array([car_model.size_car[car]['length'],\
                                     car_model.size_car[car]['width'],\
                                     car_model.size_car[car]['height']])

        rvecs.append(min_rvec.flatten())
        sizes.append(min_size.flatten())
        skeletons.append(np.array(car_model.world_car[min_car]).astype(np.float64))

    rvecs = np.array(rvecs)
    sizes = np.array(sizes)
    skeletons = np.array(skeletons)

    return dict(rvecs=rvecs, sizes=sizes, skeletons=skeletons)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str,
                        help='Path of the keypoints detection npy file.')
    parser.add_argument('--config', type=str,
                        help='Path of the config file.')

    args = parser.parse_args()

    source = args.source
    config_path = args.config
    if not os.path.exists(source):
        print('Keypoints file not exist!')
        exit(0)
    if not os.path.exists(config_path):
        print('Config file not exist!')
        exit(0)

    keypoint_data = np.load(source, allow_pickle=True)

    config = configparser.ConfigParser()
    config.read(config_path)
    K = np.array(json.loads(config['cameraIntrinsic']['cameramatrix'])).astype(
        np.float64)
    distCoeffs = np.array(json.loads(
        config['cameraIntrinsic']['distcoeffs'])).astype(np.float64)

    # Create a pool of processes, 
    # set the number of processes based on CPU core count
    num_processes = mp.cpu_count()
    pool = mp.Pool(num_processes)

    # Apply the function to each element in keypoint_data in parallel
    results = []
    with tqdm(total=len(keypoint_data)) as pbar:
        for result in pool.imap(solve, keypoint_data):
            results.append(result)
            pbar.update(1)

    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()

    # Save results
    np.save(os.path.join(OUTPUT_DIR, 'pose_shape_from_kps.npy'), results)
    print('Save pose_shape_from_kps results to {}'.format(os.path.join(OUTPUT_DIR, 'pose_shape_from_kps.npy')))