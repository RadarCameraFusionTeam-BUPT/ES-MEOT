import numpy as np
import motmetrics as mm
import sys, os
from utils.FunTools import *
import torch
from tqdm import tqdm
from itertools import permutations

es_meot_res_path = './results_all_methods/ES-MEOT/result.npy'
es_meot_no_mdn_path = './results_all_methods/ES-MEOT/result_no_mdn.npy'
es_meot_no_doped_path = './results_all_methods/ES-MEOT/result_no_doped.npy'
es_meot_no_mdn_no_doped_path = './results_all_methods/ES-MEOT/result_no_mdn_no_doped.npy'

spa_res_path = './results_all_methods/SPA/estimated_track.npy'
tpmb_bp_res_path = './results_all_methods/TPMB-BP/estimated_track.npy'
tpmbm_ca_res_path = './results_all_methods/TPMBM-CA/estimated_track.npy'
tpmbm_so_res_path = './results_all_methods/TPMBM-SO/estimated_track.npy'

bpnp_res_path = './results_all_methods/BPnP/pose_shape_from_kps_bpnp.npy'
epnp_res_path = './results_all_methods/EPnP/pose_shape_from_kps.npy'
walt_res_path = './results_all_methods/WALT3D/final_result.npy'

annotation_path = '/mnt/datasets/radar_camera_bridge/2/annotation.npy'

# Create accumulators that will be updated during each frame
acc = []
for i in range(11):
    acc.append(mm.MOTAccumulator(auto_id=True))
acc_name = ['ES-MEOT', 'ES-MEOT(no mdn)', 'ES-MEOT(no doped)', 'ES-MEOT(no mdn and doped)', 'SPA', 'TPMB-BP', 'TPMBM-CA', 'TPMBM-SO', 'BPnP', 'EPnP', 'WALT3D']

# calculate motp for distance (dis) or 1-IOU (iou)
MOTP_mean = 'dis'

es_meot_res = np.load(es_meot_res_path, allow_pickle=True)
es_meot_no_mdn_res = np.load(es_meot_no_mdn_path, allow_pickle=True)
es_meot_no_doped_res = np.load(es_meot_no_doped_path, allow_pickle=True)
es_meot_no_mdn_no_doped_res = np.load(es_meot_no_mdn_no_doped_path, allow_pickle=True)

spa_res = np.load(spa_res_path, allow_pickle=True).item()
tpmb_bp_res = np.load(tpmb_bp_res_path, allow_pickle=True).item()
tpmbm_ca_res = np.load(tpmbm_ca_res_path, allow_pickle=True).item()
tpmbm_so_res = np.load(tpmbm_so_res_path, allow_pickle=True).item()

bpnp_res = np.load(bpnp_res_path, allow_pickle=True)
epnp_res = np.load(epnp_res_path, allow_pickle=True)
walt_res = np.load(walt_res_path, allow_pickle=True)

annotation = np.load(annotation_path, allow_pickle=True)

frame_eval = 4000

theta_ellipsoid = np.linspace(0, 2*np.pi, 10)
phi_ellipsoid = np.linspace(0, np.pi, 10)
[theta_ellipsoid, phi_ellipsoid] = np.meshgrid(theta_ellipsoid, phi_ellipsoid)
theta_ellipsoid = theta_ellipsoid.flatten()
phi_ellipsoid = phi_ellipsoid.flatten()

rot_mat = np.array([np.cos(theta_ellipsoid) * np.sin(phi_ellipsoid), np.sin(theta_ellipsoid) * np.sin(phi_ellipsoid), np.cos(phi_ellipsoid)])

es_meot_r_thres = 0.5
dis_min, dis_max = 20, 120
dis_metric_thres = 4

track_length_thres = 20
tpmbm_ca_res_new = [{'id':np.array([]), 'pos':np.array([]), 'extent':np.array([])} for i in range(frame_eval)]
tpmbm_so_res_new = [{'id':np.array([]), 'pos':np.array([]), 'extent':np.array([])} for i in range(frame_eval)]

for idx, trk_ca in enumerate(tpmbm_ca_res['trajectoryEstimates']):
    ca_start_time, ca_end_time, _, ca_x_state, ca_extent_stata = trk_ca[0]
    ca_start_time = ca_start_time[0, 0] - 1
    ca_end_time = ca_end_time[0, 0] - 1

    if ca_end_time - ca_start_time > track_length_thres:
        for t in range(ca_start_time, ca_end_time+1):
            if t >= frame_eval:
                continue
            tpmbm_ca_res_new[t]['id'] = np.append(tpmbm_ca_res_new[t]['id'], idx)
            if len(tpmbm_ca_res_new[t]['pos']) == 0:
                tpmbm_ca_res_new[t]['pos'] = ca_x_state[:3, t-ca_start_time]
            else:
                tpmbm_ca_res_new[t]['pos'] = np.vstack((tpmbm_ca_res_new[t]['pos'], ca_x_state[:3, t-ca_start_time]))

            if len(tpmbm_ca_res_new[t]['extent']) == 0:
                tpmbm_ca_res_new[t]['extent'] = np.expand_dims(ca_extent_stata[:, :, t-ca_start_time], axis=0)
            else:
                tpmbm_ca_res_new[t]['extent'] = np.concatenate((tpmbm_ca_res_new[t]['extent'], np.expand_dims(ca_extent_stata[:, :, t-ca_start_time], axis=0)), axis=0)


for idx, trk_so in enumerate(tpmbm_so_res['trajectoryEstimates']):
    so_start_time, so_end_time, _, so_x_state, so_extent_stata = trk_so[0]

    so_start_time = so_start_time[0, 0] - 1
    so_end_time = so_end_time[0, 0] - 1

    if so_end_time - so_start_time > track_length_thres:
        for t in range(so_start_time, so_end_time+1):
            if t >= frame_eval:
                continue
            tpmbm_so_res_new[t]['id'] = np.append(tpmbm_so_res_new[t]['id'], idx)

            if len(tpmbm_so_res_new[t]['pos']) == 0:
                tpmbm_so_res_new[t]['pos'] = so_x_state[:3, t-so_start_time]
            else:
                tpmbm_so_res_new[t]['pos'] = np.vstack((tpmbm_so_res_new[t]['pos'], so_x_state[:3, t-so_start_time]))

            if len(tpmbm_so_res_new[t]['extent']) == 0:
                tpmbm_so_res_new[t]['extent'] = np.expand_dims(so_extent_stata[:, :, t-so_start_time], axis=0)
            else:
                tpmbm_so_res_new[t]['extent'] = np.concatenate((tpmbm_so_res_new[t]['extent'], np.expand_dims(so_extent_stata[:, :, t-so_start_time], axis=0)), axis=0)

def calculate_bbox_from_ellipsoid(extent):
    Rs, sizes = np.linalg.qr(extent)
    sizes = np.array([np.diag(s) for s in sizes])

    sizes_diag = np.array([np.diagflat(s) for s in sizes])
    sizes_sym = np.where(sizes_diag < 0, -1, np.where(sizes_diag > 0, 1, 0))

    Rs = Rs @ sizes_sym

    corner_to_extend_now = corner_to_extend.copy()
    corner_to_extend_now = corner_to_extend_now * 2
    corner_to_extend_now[:, 2] -= 1

    corner_points = corner_to_extend_now[np.newaxis, :, :] * sizes[:, np.newaxis, :]
    corner_points = corner_points @ Rs.transpose(0, 2, 1)

    return corner_points

def calculate_dims_from_ellipsoid(extent):
    Rs, sizes = np.linalg.qr(extent)
    sizes = np.array([np.diag(s) for s in sizes])

    return sizes

def calculate_iou(dim1, dim2):
    intersection = np.prod(np.minimum(dim1, dim2))
    union = np.prod(dim1) + np.prod(dim2) - intersection
    iou = intersection / union
    return iou

def best_iou(detected_dims, true_dims):
    best_iou_value = 0
    best_permutation = None
    
    for perm in permutations(detected_dims):
        iou_value = calculate_iou(np.array(perm), np.array(true_dims))
        if iou_value > best_iou_value:
            best_iou_value = iou_value
            best_permutation = perm
    
    return best_iou_value, best_permutation


# relationship of corners and extend
corner_to_extend = np.array([
    [0.5, -0.5, 0],
    [-0.5, -0.5, 0],
    [-0.5, 0.5, 0],
    [0.5, 0.5, 0],
    [0.5, -0.5, 1],
    [-0.5, -0.5, 1],
    [-0.5, 0.5, 1],
    [0.5, 0.5, 1]
])

# fix rotation for WALT3D
R_fix = np.array([[ 0.98768834, -0.01570003,  0.15564463],
                [ 0.,          0.99495102,  0.10036171],
                [-0.15643447, -0.0991261,   0.98270152]])
T_fix = np.array([0, 1, -4])

for frame_id in tqdm(range(frame_eval), total=frame_eval):
    #### gt results ####
    gt_r = np.linalg.norm(annotation[frame_id]['pos_bottom_mid'], axis=1)
    gt_mask = np.logical_and(gt_r <= dis_max, gt_r >= dis_min)

    gt_ids = annotation[frame_id]['ids'][gt_mask].astype(int)

    gt_pos = annotation[frame_id]['pos_bottom_mid'][gt_mask]
    gt_theta = annotation[frame_id]['rvecs'][gt_mask]
    gt_R = rotation_vectors_to_matrices(gt_theta)
    gt_ext = annotation[frame_id]['sizes'][gt_mask]

    gt_corner_point = corner_to_extend[np.newaxis, :, :] * gt_ext[:, np.newaxis, :]
    gt_corner_world = gt_pos[:, np.newaxis, :] + gt_corner_point @ gt_R.transpose(0, 2, 1)

    #### ES_EOT detections ####
    es_meot_r = np.linalg.norm(es_meot_res[frame_id]['x_ref'][:, :3], axis=1)
    es_meot_mask = np.logical_and(es_meot_res[frame_id]['r'] > es_meot_r_thres, np.logical_and(es_meot_r <= dis_max, es_meot_r >= dis_min))

    es_meot_ids = (es_meot_res[frame_id]['t'][es_meot_mask] * 1000 + es_meot_res[frame_id]['t_id'][es_meot_mask]).astype(int)

    es_meot_pos = es_meot_res[frame_id]['x_ref'][es_meot_mask, :3]

    if MOTP_mean == 'iou':
        pass
        # es_meot_theta = es_meot_res[frame_id]['x_ref'][es_meot_mask, 4:7]
        # es_meot_R = rotation_vectors_to_matrices(es_meot_theta)
        # es_meot_ext = es_meot_res[frame_id]['x_ref'][es_meot_mask, 10:13]

        # es_meot_corner_point = corner_to_extend[np.newaxis, :, :] * es_meot_ext[:, np.newaxis, :]
        # es_meot_corner_world = es_meot_pos[:, np.newaxis, :] + es_meot_corner_point @ es_meot_R.transpose(0, 2, 1)

        # intersection_vol, iou_3d = box3d_overlap(torch.from_numpy(gt_corner_world).to(torch.float32), torch.from_numpy(es_meot_corner_world).to(torch.float32))

        # dis = 1 - iou_3d.numpy()
        # dis[dis == 1.0] = np.nan

    elif MOTP_mean == 'dis':
        dis = np.linalg.norm(gt_pos[:, np.newaxis, :] - es_meot_pos[np.newaxis, :, :], axis=-1)
        dis[dis>dis_metric_thres] = np.nan

    # record in acc
    acc[0].update(gt_ids.tolist(), es_meot_ids.tolist(), dis.tolist())

    #### ES_EOT no mdn detections ####
    es_meot_no_mdn_r = np.linalg.norm(es_meot_no_mdn_res[frame_id]['x_ref'][:, :3], axis=1)
    es_meot_no_mdn_mask = np.logical_and(es_meot_no_mdn_res[frame_id]['r'] > es_meot_r_thres, np.logical_and(es_meot_no_mdn_r <= dis_max, es_meot_no_mdn_r >= dis_min))

    es_meot_no_mdn_ids = (es_meot_no_mdn_res[frame_id]['t'][es_meot_no_mdn_mask] * 1000 + es_meot_no_mdn_res[frame_id]['t_id'][es_meot_no_mdn_mask]).astype(int)

    es_meot_no_mdn_pos = es_meot_no_mdn_res[frame_id]['x_ref'][es_meot_no_mdn_mask, :3]

    dis = np.linalg.norm(gt_pos[:, np.newaxis, :] - es_meot_no_mdn_pos[np.newaxis, :, :], axis=-1)
    dis[dis>dis_metric_thres] = np.nan

    # record in acc
    acc[1].update(gt_ids.tolist(), es_meot_no_mdn_ids.tolist(), dis.tolist())

    #### ES_EOT no doped detections ####
    es_meot_no_doped_r = np.linalg.norm(es_meot_no_doped_res[frame_id]['x_ref'][:, :3], axis=1)
    es_meot_no_doped_mask = np.logical_and(es_meot_no_doped_res[frame_id]['r'] > es_meot_r_thres, np.logical_and(es_meot_no_doped_r <= dis_max, es_meot_no_doped_r >= dis_min))

    es_meot_no_doped_ids = (es_meot_no_doped_res[frame_id]['t'][es_meot_no_doped_mask] * 1000 + es_meot_no_doped_res[frame_id]['t_id'][es_meot_no_doped_mask]).astype(int)

    es_meot_no_doped_pos = es_meot_no_doped_res[frame_id]['x_ref'][es_meot_no_doped_mask, :3]

    dis = np.linalg.norm(gt_pos[:, np.newaxis, :] - es_meot_no_doped_pos[np.newaxis, :, :], axis=-1)
    dis[dis>dis_metric_thres] = np.nan

    # record in acc
    acc[2].update(gt_ids.tolist(), es_meot_no_doped_ids.tolist(), dis.tolist())

    #### ES_EOT no mdn no doped detections ####
    es_meot_no_mdn_no_doped_r = np.linalg.norm(es_meot_no_mdn_no_doped_res[frame_id]['x_ref'][:, :3], axis=1)
    es_meot_no_mdn_no_doped_mask = np.logical_and(es_meot_no_mdn_no_doped_res[frame_id]['r'] > es_meot_r_thres, np.logical_and(es_meot_no_mdn_no_doped_r <= dis_max, es_meot_no_mdn_no_doped_r >= dis_min))

    es_meot_no_mdn_no_doped_ids = (es_meot_no_mdn_no_doped_res[frame_id]['t'][es_meot_no_mdn_no_doped_mask] * 1000 + es_meot_no_mdn_no_doped_res[frame_id]['t_id'][es_meot_no_mdn_no_doped_mask]).astype(int)

    es_meot_no_mdn_no_doped_pos = es_meot_no_mdn_no_doped_res[frame_id]['x_ref'][es_meot_no_mdn_no_doped_mask, :3]

    dis = np.linalg.norm(gt_pos[:, np.newaxis, :] - es_meot_no_mdn_no_doped_pos[np.newaxis, :, :], axis=-1)
    dis[dis>dis_metric_thres] = np.nan

    # record in acc
    acc[3].update(gt_ids.tolist(), es_meot_no_mdn_no_doped_ids.tolist(), dis.tolist())

    #### SPA detections ####
    spa_pos = spa_res['estimatedTracks'][:, frame_id, :]
    spa_ext = spa_res['estimatedExtents'][:, :, frame_id, :]

    spa_mask = np.all(~np.isnan(spa_pos), axis=0)
    spa_r = np.linalg.norm(spa_pos[:3, :], axis=0)
    spa_mask = np.logical_and(spa_mask, np.logical_and(spa_r <= dis_max, spa_r >= dis_min))

    spa_pos = spa_pos[:3, spa_mask].T
    
    spa_ids = np.where(spa_mask)[0]
    # spa_extent = calculate_bbox_from_ellipsoid(spa_ext[:, :, spa_mask].transpose(2, 0, 1))
    # spa_world = spa_extent + spa_pos[:, np.newaxis, :]

    # intersection_vol, iou_3d = box3d_overlap(torch.from_numpy(gt_corner_world).to(torch.float32), torch.from_numpy(spa_world).to(torch.float32))
    # iou = 1 - iou_3d.numpy()
    # iou[iou == 1.0] = np.nan

    dis = np.linalg.norm(gt_pos[:, np.newaxis, :] - spa_pos[np.newaxis, :, :], axis=-1)
    dis[dis>dis_metric_thres] = np.nan

    # record in acc
    acc[4].update(gt_ids.tolist(), spa_ids.tolist(), dis.tolist())

    #### TPMB-BP detections ####
    tpmb_bp_pos = tpmb_bp_res['estimatedTracks'][:, frame_id, :]
    tpmb_bp_ext = tpmb_bp_res['estimatedExtents'][:, :, frame_id, :]

    tpmb_bp_mask = np.all(~np.isnan(tpmb_bp_pos), axis=0)
    tpmb_bp_r = np.linalg.norm(tpmb_bp_pos[:3, :], axis=0)
    tpmb_bp_mask = np.logical_and(tpmb_bp_mask, np.logical_and(tpmb_bp_r <= dis_max, tpmb_bp_r >= dis_min))

    tpmb_bp_pos = tpmb_bp_pos[:3, tpmb_bp_mask].T
    
    tpmb_bp_ids = np.where(tpmb_bp_mask)[0]

    dis = np.linalg.norm(gt_pos[:, np.newaxis, :] - tpmb_bp_pos[np.newaxis, :, :], axis=-1)
    dis[dis>dis_metric_thres] = np.nan

    # record in acc
    acc[5].update(gt_ids.tolist(), tpmb_bp_ids.tolist(), dis.tolist())

    #### TPMBM-CA detections ####
    tpmbm_ca_pos = tpmbm_ca_res_new[frame_id]['pos']
    tpmbm_ca_ext = tpmbm_ca_res_new[frame_id]['extent']
    tpmbm_ca_id = tpmbm_ca_res_new[frame_id]['id']

    tpmbm_ca_r = np.linalg.norm(tpmbm_ca_pos, axis=1)
    tpmbm_ca_mask = np.logical_and(tpmbm_ca_r <= dis_max, tpmbm_ca_r >= dis_min)

    dis = np.linalg.norm(gt_pos[:, np.newaxis, :] - tpmbm_ca_pos[np.newaxis, tpmbm_ca_mask, :], axis=-1)
    dis[dis>dis_metric_thres] = np.nan

    # record in acc
    acc[6].update(gt_ids.tolist(), tpmbm_ca_id[tpmbm_ca_mask].astype(int).tolist(), dis.tolist())

    #### TPMBM-SO detections ####
    tpmbm_so_pos = tpmbm_so_res_new[frame_id]['pos']
    tpmbm_so_ext = tpmbm_so_res_new[frame_id]['extent']
    tpmbm_so_id = tpmbm_so_res_new[frame_id]['id']

    tpmbm_so_r = np.linalg.norm(tpmbm_so_pos, axis=1)
    tpmbm_so_mask = np.logical_and(tpmbm_so_r <= dis_max, tpmbm_so_r >= dis_min)

    dis = np.linalg.norm(gt_pos[:, np.newaxis, :] - tpmbm_so_pos[np.newaxis, tpmbm_so_mask, :], axis=-1)
    dis[dis>dis_metric_thres] = np.nan

    # record in acc
    acc[7].update(gt_ids.tolist(), tpmbm_so_id[tpmbm_so_mask].astype(int).tolist(), dis.tolist())

    #### BPnP results ####
    bpnp_pos = bpnp_res[frame_id]['translations']
    bpnp_ext = bpnp_res[frame_id]['sizes']
    bpnp_id = bpnp_res[frame_id]['ids']

    bpnp_r = np.linalg.norm(bpnp_pos, axis=1)
    bpnp_mask = np.logical_and(bpnp_r <= dis_max, bpnp_r >= dis_min)

    dis = np.linalg.norm(gt_pos[:, np.newaxis, :] - bpnp_pos[np.newaxis, bpnp_mask, :], axis=-1)
    dis[dis>dis_metric_thres] = np.nan

    # record in acc
    acc[8].update(gt_ids.tolist(), bpnp_id[bpnp_mask].astype(int).tolist(), dis.tolist())

    #### EPnP results ####
    epnp_pos = epnp_res[frame_id]['tvecs']
    epnp_ext = epnp_res[frame_id]['sizes']
    epnp_id = epnp_res[frame_id]['ids']

    epnp_r = np.linalg.norm(epnp_pos, axis=1)
    epnp_mask = np.logical_and(epnp_r <= dis_max, epnp_r >= dis_min)

    dis = np.linalg.norm(gt_pos[:, np.newaxis, :] - epnp_pos[np.newaxis, epnp_mask, :], axis=-1)
    dis[dis>dis_metric_thres] = np.nan

    # record in acc
    acc[9].update(gt_ids.tolist(), epnp_id[epnp_mask].astype(int).tolist(), dis.tolist())

    #### WALT3D results ####
    walt_pos = walt_res[frame_id]['translation']
    walt_id = walt_res[frame_id]['ids']
    walt_pos_fix = walt_pos @ R_fix.T + T_fix

    walt_r = np.linalg.norm(walt_pos_fix, axis=1)
    walt_mask = np.logical_and(walt_r <= dis_max, walt_r >= dis_min)

    dis = np.linalg.norm(gt_pos[:, np.newaxis, :] - walt_pos_fix[np.newaxis, walt_mask, :], axis=-1)
    dis[dis>dis_metric_thres] = np.nan

    # record in acc
    acc[10].update(gt_ids.tolist(), walt_id[walt_mask].astype(int).tolist(), dis.tolist())


mh = mm.metrics.create()

for i, a in enumerate(acc):
    summary = mh.compute(a, metrics=['num_frames', 'mota', 'motp', 'num_false_positives', \
                                    'num_misses', 'num_switches', 'mostly_tracked', \
                                    'mostly_lost', 'idp', 'idr', 'idf1' \
                                    ], \
                        name=acc_name[i])

    strsummary = mm.io.render_summary(
        summary,
        #formatters={'mota' : '{:.2%}'.format},
        namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll', \
                'precision': 'Prcn', 'num_objects': 'GT', \
                'mostly_tracked' : 'MT', 'partially_tracked': 'PT', \
                'mostly_lost' : 'ML', 'num_false_positives': 'FP', \
                'num_misses': 'FN', 'num_switches' : 'IDsw', \
                'num_fragmentations' : 'FM', 'mota': 'MOTA', 'motp' : 'MOTP',  \
                }
    )
    print(strsummary)