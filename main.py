import argparse
import os
import torch
import numpy as np
from utils.parameters import N_T, lambda_b
from utils.TPMB_RCfusion import TPMB_RCfusion_predict, TPMB_RCfusion_update, TPMB_RCfusion_prune, State_trk, MultiBern, PPP
from utils.Space2Plane import Space2Plane
from utils.network_model import MDN
from tqdm import tqdm

ROOT_DIR = os.path.abspath("./")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataPath", help='Folder which include radarPcl.npy, output-keypoints.npy, pose_shape_from_kps, mdn.pt and config', nargs=1, type=str)
    args = parser.parse_args()
    DATA_PATH = args.dataPath[0]

    ######## load data ##########
    if not os.path.exists('{}/radarPcl.npy'.format(DATA_PATH)):
        print('{}/radarPcl.npy not exist!'.format(DATA_PATH))
        exit(0)
    if not os.path.exists('{}/output-keypoints.npy'.format(DATA_PATH)):
        print('{}/output-keypoints.npy not exist!'.format(DATA_PATH))
        exit(0)
    if not os.path.exists('{}/pose_shape_from_kps.npy'.format(DATA_PATH)):
        print('{}/pose_shape_from_kps.npy not exist!'.format(DATA_PATH))
        exit(0)
    if not os.path.exists('{}/config'.format(DATA_PATH)):
        print('{}/config not exist!'.format(DATA_PATH))
        exit(0)
    if not os.path.exists('{}/mdn.pt'.format(DATA_PATH)):
        print('{}/mdn.pt not exist!'.format(DATA_PATH))
        exit(0)

    radarPcl = np.load('{}/radarPcl.npy'.format(DATA_PATH), allow_pickle=True)
    keypoints = np.load(
        '{}/output-keypoints.npy'.format(DATA_PATH), allow_pickle=True)
    pose_shape = np.load(
        '{}/pose_shape_from_kps.npy'.format(DATA_PATH), allow_pickle=True)
    config_path = os.path.join(DATA_PATH, 'config')
    s2p = Space2Plane(config=config_path)
    mdn = torch.load('{}/mdn.pt'.format(DATA_PATH))

    measurements = [[[pcl['x'], pcl['y'], pcl['z']]
                     for pcl in frame['data'] if pcl['doppler'] != 0] for frame in radarPcl]
    pixel_keypoints = [frame['keypoints'] for frame in keypoints]

    ####### initialize birth model ######
    birth = PPP()

    # Initialize birth.trks_state
    np.random.seed(42)
    mu = np.random.normal(0, 1, (N_T, 9))
    mu[:, 6:] = 0

    Sigma = np.tile(np.identity(9), (N_T, 1, 1))
    birth.trks_state = [
        State_trk(
            x_ref=np.array([0, 0, 170, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]).astype(float),
            dx=np.zeros(13),
            P=np.eye(13) * 10,
            mu=mu,
            Sigma=Sigma
        )
    ]
    n_birth = len(birth.trks_state)

    # Initialize birth.weights
    birth.weights = np.ones(n_birth) * lambda_b

    # Initialize birth.trks_t
    birth.trks_t = np.zeros(n_birth)

    ####### initialize PPP and MB ######
    ppp = PPP()

    mb = MultiBern()

    ############ main loop ############
    frame_s, frame_e = 0, 4000
    res = []
    for i, (pcl, kps, pose_shape_frame) in tqdm(enumerate(zip(measurements[frame_s:frame_e], pixel_keypoints[frame_s:frame_e], pose_shape[frame_s:frame_e])), total=frame_e-frame_s):
        now = dict()
        ppp, mb = TPMB_RCfusion_predict(ppp, mb, birth)
        birth.trks_t += 1

        # transfer pcl to camera coordinate system
        pcl = s2p.radar2cameraXYZ(np.asarray(pcl).reshape([-1, 3]))
        ppp, mb = TPMB_RCfusion_update(ppp, mb, pcl, kps, pose_shape_frame, mdn, s2p)
        ppp, mb = TPMB_RCfusion_prune(ppp, mb)

        if len(mb.trks_r) > 0:
            now['r'] = mb.trks_r.copy()
            now['t'] = mb.trks_t.copy()
            now['t_id'] = mb.trks_t_id.copy()
            now['x_ref'] = np.stack([trk.trk_x_ref[-1] for trk in mb.trks_state], axis=0)
            now['P'] = np.stack([trk.trk_P[-1] for trk in mb.trks_state], axis=0)
            now['mu'] = np.stack([trk.trk_mu[-1] for trk in mb.trks_state], axis=0)
            now['Sigma'] = np.stack([trk.trk_Sigma[-1] for trk in mb.trks_state], axis=0)
        else:
            now['r'] = np.array([])
            now['t'] = np.array([])
            now['t_id'] = np.array([])
            now['x_ref'] = np.array([])
            now['P'] = np.array([])
            now['mu'] = np.array([])
            now['Sigma'] = np.array([])
        res.append(now)

    np.save(os.path.join(OUTPUT_DIR,\
        'result.npy'), res, allow_pickle = True)