from .parameters import *
from .FunTools import *
import copy
import torch
from sklearn.mixture import GaussianMixture
from murty import Murty
from sklearn.cluster import DBSCAN
from . import car_model
import cv2


class State_trk:
    def __init__(self, x_ref, dx, P, mu, Sigma):
        """
        Initialize a State_trk object.

        Parameters:
        - x_ref (array-like, n x 13): Reference state (x, y, z, v, theta_x, theta_y, theta_z, omega_x, omega_y, omega_z, length, width, height).
        - dx (array-like, n x 13): Error-state (dx, dy, dz, dv, d_theta_x, d_theta_y, d_theta_z, d_omega_x, d_omega_y, d_omega_z, d_length, d_width, d_height).
        - P (array-like, n x 13 x 13): Covariance matrix.
        - mu (array-like, n x N_T x 9): State of reflection points.
        - Sigma (array-like, n x N_T x 9 x 9): Covariance matrix of reflection points.
        """
        self.trk_x_ref = x_ref if x_ref.ndim == 2 else np.expand_dims(x_ref, axis=0)
        self.trk_dx = dx if dx.ndim == 2 else np.expand_dims(dx, axis=0)
        self.trk_P = P if P.ndim == 3 else np.expand_dims(P, axis=0)
        self.trk_mu = mu if mu.ndim == 3 else np.expand_dims(mu, axis=0)
        self.trk_Sigma = Sigma if Sigma.ndim == 4 else np.expand_dims(Sigma, axis=0)

        assert self.trk_x_ref.shape[1] == 13 and self.trk_dx.shape[1] == 13, 'x_ref or dx dim error'
        assert self.trk_P.shape[1] == 13 and self.trk_P.shape[2] == 13, 'P dim error'
        assert self.trk_mu.shape[2] == 9, 'mu dim error'
        assert self.trk_Sigma.shape[2] == 9 and self.trk_Sigma.shape[3] == 9, 'Sigma dim error'


class MultiBern:
    def __init__(self):
        self.trks_r = np.array([])
        self.trks_t = np.array([])
        self.trks_t_id = np.array([])
        self.trks_state = []

    def predict(self):
        self.trks_r *= Ps

        if len(self.trks_state) > 0:
            # stack all the state that need to be predicted
            trks_x_ref_con = np.stack(
                [trk.trk_x_ref[-1] for trk in self.trks_state], axis=0)
            trks_P_con = np.stack(
                [trk.trk_P[-1] for trk in self.trks_state], axis=0)
            trks_mu_con = np.stack(
                [trk.trk_mu[-1] for trk in self.trks_state], axis=0)
            trks_Sigma_con = np.stack(
                [trk.trk_Sigma[-1] for trk in self.trks_state], axis=0)

            # predict next state of every track
            trks_x_ref_con, trks_P_con, trks_mu_con, trks_Sigma_con = predict_batch(
                trks_x_ref_con, trks_P_con, trks_mu_con, trks_Sigma_con)

            # add predictions on trks_state
            for i, trk in enumerate(self.trks_state):
                trk.trk_x_ref = np.vstack((trk.trk_x_ref, trks_x_ref_con[i:i+1]))
                trk.dx = np.vstack((trk.trk_dx, np.zeros((1, 13))))
                trk.trk_P = np.vstack((trk.trk_P, trks_P_con[i:i+1]))
                trk.trk_mu = np.vstack((trk.trk_mu, trks_mu_con[i:i+1]))
                trk.trk_Sigma = np.vstack((trk.trk_Sigma, trks_Sigma_con[i:i+1]))

    def update(self, ppp, pcl, kps, pose_shape, mdn, s2p):
        pcl = np.asarray(pcl).reshape([-1, 3])
        corner_kps = np.zeros((0, 2))
        if len(kps) > 0:
            remain_idx = [i for i, kp in enumerate(kps) if len(kp) > 0 and len(np.asarray(kp)[np.isin(np.asarray(kp)[:, 0], corner_id)]) == 8 and len(np.where(np.asarray(kp)[:, 0] < 24)[0]) > 2]

            pose_shape['rvecs'] = pose_shape['rvecs'][remain_idx]
            pose_shape['sizes'] = pose_shape['sizes'][remain_idx]
            pose_shape['skeletons'] = pose_shape['skeletons'][remain_idx]

            kps = [kps[i] for i in remain_idx]
            corner_kps = np.array([np.asarray(kp)[np.isin(np.asarray(kp)[:, 0], corner_id), 1:] for kp in kps])

            # calculate estimated bottom middle from mdn
            bottom_middle_pix = np.array([CalBottomMiddle(kp) for kp in kps])
            if len(bottom_middle_pix) == 0:
                bottom_middle_pix = np.zeros((0, 2))

            pi, mu, sigma = mdn(torch.from_numpy(bottom_middle_pix.copy()))
            pi = pi.detach().numpy()
            mu = mu.detach().numpy()
            sigma = sigma.detach().numpy()
            
            bottom_middle_P = [np.diag(item) for item in np.sum(pi[..., np.newaxis] * sigma, axis=1)]

            #### Ablation 1 ####
            ## with MDN
            bottom_middle_pos = np.sum(pi[..., np.newaxis] * mu, axis=1)

            ## without MDN
            # def back_proj(pix):
            #     a, b, c = road_vector
            #     d = d_ground
            #     fx, fy = s2p.cameraMatrix[0, 0], s2p.cameraMatrix[1, 1]
            #     u0, v0 = s2p.cameraMatrix[0, 2], s2p.cameraMatrix[1, 2]
            #     z = -d / (a*(pix[:, 0]-u0)/fx + b*(pix[:, 1]-v0)/fy + c)
            #     x = (pix[:, 0]-u0)*z/fx
            #     y = (pix[:, 1]-v0)*z/fy

            #     return np.stack((x, y, z), axis=1)
            # bottom_middle_pos = back_proj(bottom_middle_pix)
        
        n_kps = len(corner_kps)

        ####### measurements partition #######
        Z_p_c, s_i, Z_p_cor_pix, Z_p_pose, Z_p_shape, Z_p_skeleton, Z_p_middle_pos, Z_p_middle_P, valid_r, Z_p_r, cluster_typ = partition_based_on_projection(pcl, kps, corner_kps, pose_shape, bottom_middle_pos, bottom_middle_P, s2p)

        n_new_bern = len(Z_p_c)

        # calculate log likelihood of exist bern towards partitions
        n_exist_bern = len(self.trks_state)
        
        ####### calculate cost matrix #######
        cost_mat = np.full((n_new_bern, n_exist_bern + n_new_bern), np.inf)
        cost_thres = np.full((n_new_bern, n_exist_bern + n_new_bern), np.inf)

        if n_exist_bern > 0:
            # stack all the state that need to be updated
            trks_x_ref_con = np.stack(
                [trk.trk_x_ref[-1] for trk in self.trks_state], axis=0)
            trks_P_con = np.stack(
                [trk.trk_P[-1] for trk in self.trks_state], axis=0)
            trks_mu_con = np.stack(
                [trk.trk_mu[-1] for trk in self.trks_state], axis=0)
            trks_Sigma_con = np.stack(
                [trk.trk_Sigma[-1] for trk in self.trks_state], axis=0)

            # updated results for every exist bern 
            # towards every partition at the same time
            trks_x_ref_expand = np.tile(trks_x_ref_con, (n_new_bern, 1))
            trks_dx_expand = np.zeros_like(trks_x_ref_expand)
            trks_P_expand = np.tile(trks_P_con, (n_new_bern, 1, 1))
            trks_mu_expand = np.tile(trks_mu_con, (n_new_bern, 1, 1))
            trks_Sigma_expand = np.tile(trks_Sigma_con, (n_new_bern, 1, 1, 1))
            clus_id_expand = np.kron(np.arange(n_new_bern), np.ones(n_exist_bern)).astype(int)

            trks_x_ref_update_all = np.zeros((n_new_bern * n_exist_bern, 13))
            trks_P_update_all = np.zeros((n_new_bern * n_exist_bern, 13, 13))
            trks_mu_update_all = np.zeros((n_new_bern * n_exist_bern, N_T, 9))
            trks_Sigma_update_all = np.zeros((n_new_bern * n_exist_bern, N_T, 9, 9))
            log_likelihood_update_all = np.zeros(n_new_bern * n_exist_bern)
            likeli_thres_update_all = np.zeros(n_new_bern * n_exist_bern)

            for typ in [0, 1, 2]:
                mask = np.isin(clus_id_expand, np.where(cluster_typ == typ))
                if not np.any(mask):
                    continue
                trks_x_ref_update_all[mask], trks_P_update_all[mask], trks_mu_update_all[mask], trks_Sigma_update_all[mask], log_likelihood_update_all[mask], likeli_thres_update_all[mask] = update_using_measurements(
                    trks_x_ref_expand[mask], trks_dx_expand[mask], trks_P_expand[mask], trks_mu_expand[mask], trks_Sigma_expand[mask], clus_id_expand[mask], Z_p_r, valid_r, Z_p_c, s_i, Z_p_cor_pix, Z_p_pose, Z_p_shape, Z_p_skeleton, Z_p_middle_pos, Z_p_middle_P, typ, s2p)
            
            cost_thres[:, :n_exist_bern] = likeli_thres_update_all.reshape([n_new_bern, n_exist_bern])
            cost_mat[:, :n_exist_bern] = -log_likelihood_update_all.reshape([n_new_bern, n_exist_bern])

        if n_new_bern:
            # calculate log likelihood of new bern towards partitions
            # stack all the PPP state
            ppp_x_ref_con = np.stack(
                [trk.trk_x_ref[-1] for trk in ppp.trks_state], axis=0)
            ppp_P_con = np.stack(
                [trk.trk_P[-1] for trk in ppp.trks_state], axis=0)
            ppp_mu_con = np.stack(
                [trk.trk_mu[-1] for trk in ppp.trks_state], axis=0)
            ppp_Sigma_con = np.stack(
                [trk.trk_Sigma[-1] for trk in ppp.trks_state], axis=0)
            n_ppp = len(ppp_x_ref_con)

            # updated results for every ppp component 
            # towards every partition at the same time
            ppp_x_ref_expand = np.tile(ppp_x_ref_con, (n_new_bern, 1))
            ppp_dx_expand = np.zeros_like(ppp_x_ref_expand)
            ppp_P_expand = np.tile(ppp_P_con, (n_new_bern, 1, 1))
            ppp_mu_expand = np.tile(ppp_mu_con, (n_new_bern, 1, 1))
            ppp_Sigma_expand = np.tile(ppp_Sigma_con, (n_new_bern, 1, 1, 1))
            ppp_clus_id_expand = np.kron(np.arange(n_new_bern), np.ones(n_ppp)).astype(int)

            ppp_x_ref_update_all = np.zeros((n_new_bern * n_ppp, 13))
            ppp_P_update_all = np.zeros((n_new_bern * n_ppp, 13, 13))
            ppp_mu_update_all = np.zeros((n_new_bern * n_ppp, N_T, 9))
            ppp_Sigma_update_all = np.zeros((n_new_bern * n_ppp, N_T, 9, 9))
            ppp_log_likelihood_update_all = np.zeros(n_new_bern * n_ppp)
            ppp_likeli_thres_update_all = np.zeros(n_new_bern * n_ppp)
                        
            for typ in [0, 1, 2]:
                mask = np.isin(ppp_clus_id_expand, np.where(cluster_typ == typ))
                if not np.any(mask):
                    continue
                ppp_x_ref_update_all[mask], ppp_P_update_all[mask], ppp_mu_update_all[mask], ppp_Sigma_update_all[mask], ppp_log_likelihood_update_all[mask], ppp_likeli_thres_update_all[mask] = update_using_measurements(
                    ppp_x_ref_expand[mask], ppp_dx_expand[mask], ppp_P_expand[mask], ppp_mu_expand[mask], ppp_Sigma_expand[mask], ppp_clus_id_expand[mask], Z_p_r, valid_r, Z_p_c, s_i, Z_p_cor_pix, Z_p_pose, Z_p_shape, Z_p_skeleton, Z_p_middle_pos, Z_p_middle_P, typ, s2p)

            ppp_x_ref_update_all_reshape = ppp_x_ref_update_all.reshape([n_new_bern, n_ppp, 13])
            ppp_P_update_all_reshape = ppp_P_update_all.reshape([n_new_bern, n_ppp, 13, 13])
            ppp_mu_update_all_reshape = ppp_mu_update_all.reshape([n_new_bern, n_ppp, N_T, 9])
            ppp_Sigma_update_all_reshape = ppp_Sigma_update_all.reshape([n_new_bern, n_ppp, N_T, 9, 9])
            
            ppp_log_likelihood_update_all_reshape = ppp_log_likelihood_update_all.reshape([n_new_bern, n_ppp])
            ppp_likeli_thres_update_all_reshape = ppp_likeli_thres_update_all.reshape([n_new_bern, n_ppp])

            ppp_weight_sum = np.sum(ppp.weights)

            # log likelihood of new bern
            # by weighted average of ppp components
            C = np.max(ppp_log_likelihood_update_all_reshape, axis=1, keepdims=True)
            new_log_likelihood_update = (C + np.log(np.sum(ppp.weights[np.newaxis, :] * np.exp(ppp_log_likelihood_update_all_reshape-C), axis=1, keepdims=True)/ppp_weight_sum)).squeeze(axis=-1)

            C = np.max(ppp_likeli_thres_update_all_reshape, axis=1, keepdims=True)
            new_log_likelihood_thres_update = (C + np.log(np.sum(ppp.weights[np.newaxis, :] * np.exp(ppp_likeli_thres_update_all_reshape-C), axis=1, keepdims=True)/ppp_weight_sum)).squeeze(axis=-1)

            # new bern state and weights
            new_x_ref_update = np.sum(ppp_x_ref_update_all_reshape * ppp.weights[np.newaxis, :, np.newaxis], axis=1) / ppp_weight_sum
            new_P_update = np.sum(ppp_P_update_all_reshape * ppp.weights[np.newaxis, :, np.newaxis, np.newaxis], axis=1) / np.square(ppp_weight_sum)
            new_mu_update = np.sum(ppp_mu_update_all_reshape * ppp.weights[np.newaxis, :, np.newaxis, np.newaxis], axis=1) / ppp_weight_sum
            new_Sigma_update = np.sum(ppp_Sigma_update_all_reshape * ppp.weights[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis], axis=1) / np.square(ppp_weight_sum)

            new_log_weight_update = ppp_log_likelihood_update_all_reshape + np.log(ppp.weights)
            C = np.max(new_log_weight_update, axis=1)
            new_log_weight_update -= C[:, np.newaxis]
            new_log_weight_update = np.log(np.sum(np.exp(new_log_weight_update), axis=1)) + C

            # put the log likelihood to cost matrix
            cost_mat[np.arange(n_new_bern), n_exist_bern+np.arange(n_new_bern)] = -new_log_likelihood_update
            cost_thres[np.arange(n_new_bern), n_exist_bern+np.arange(n_new_bern)] = new_log_likelihood_thres_update

        ####### generate assignment matrix #######
        # remove low likelihood associations according to threshold
        cost_mat[cost_mat > cost_thres] = np.inf
        try:
            max_non_inf = np.max(cost_mat[cost_mat != np.inf])
        except ValueError:
            max_non_inf = 1.0
        cost_mat[np.arange(n_new_bern), n_exist_bern+np.arange(n_new_bern)] = max_non_inf

        # calculate assignment matrix
        assignments = np.full((n_exist_bern + n_new_bern, M), -1).astype(int)

        # murty
        # cost: row is new bern, col is exist bern + new bern
        # sol: associated id for new bern
        # M_new: number of efficient hypothesis
        M_new = 0
        if len(cost_mat) != 0:
            mgen = Murty(cost_mat)
            for i in range(M):
                ok, _, sol = mgen.draw()
                if ok:
                    M_new += 1
                    assignments[sol, i] = np.arange(n_new_bern)

        ####### update exist bern #######
        log_weight_hyp_exist = np.zeros((n_exist_bern, M_new))
        trks_r_hyp_exist = np.zeros((n_exist_bern, M_new))

        trks_x_ref_hyp_exist = np.zeros((n_exist_bern, M_new, 13))
        trks_P_hyp_exist = np.zeros((n_exist_bern, M_new, 13, 13))
        trks_mu_hyp_exist = np.zeros((n_exist_bern, M_new, N_T, 9))
        trks_Sigma_hyp_exist = np.zeros((n_exist_bern, M_new, N_T, 9, 9))

        if n_exist_bern > 0:
            # temporary variables
            assign_exist = assignments[:n_exist_bern, :M_new]
            trks_r_ori = np.stack([self.trks_r] * M_new, axis=1)
            trks_x_ref_ori = np.stack([trks_x_ref_con] * M_new, axis=1)
            trks_dx_ori = np.zeros((n_exist_bern, M_new, 13))
            trks_P_ori = np.stack([trks_P_con] * M_new, axis=1)
            trks_mu_ori = np.stack([trks_mu_con] * M_new, axis=1)
            trks_Sigma_ori = np.stack([trks_Sigma_con] * M_new, axis=1)

            # update miss detected tracks
            mask_miss = assign_exist == -1
            if np.any(mask_miss):
                trks_r_hyp_exist[mask_miss] = trks_r_ori[mask_miss] * q_d_c * q_d_r / (
                    1 - trks_r_ori[mask_miss] + trks_r_ori[mask_miss] * q_d_c * q_d_r)
                log_weight_hyp_exist[mask_miss] = np.log(
                    1 - trks_r_ori[mask_miss] + trks_r_ori[mask_miss] * q_d_c * q_d_r)

                trks_x_ref_hyp_exist[mask_miss] = trks_x_ref_ori[mask_miss]
                trks_P_hyp_exist[mask_miss] = trks_P_ori[mask_miss]
                trks_mu_hyp_exist[mask_miss] = trks_mu_ori[mask_miss]
                trks_Sigma_hyp_exist[mask_miss] = trks_Sigma_ori[mask_miss]
                
            # update with only radar measurements (typ=0) 
            # only keypoints measurements (typ=1)
            # both radar and keypoints measurements (typ=2)
            for typ in [0, 1, 2]:
                mask = np.isin(assign_exist, np.where(cluster_typ == typ))
                if not np.any(mask):
                    continue
                bern_id, hyp_id = np.where(mask)
                partition_id = assign_exist[bern_id, hyp_id]

                trks_r_hyp_exist[mask] = 1.0
                trks_x_ref_hyp_exist[mask] = trks_x_ref_update_all[partition_id * n_exist_bern + bern_id]
                trks_P_hyp_exist[mask] = trks_P_update_all[partition_id * n_exist_bern + bern_id]
                trks_mu_hyp_exist[mask] = trks_mu_update_all[partition_id * n_exist_bern + bern_id]
                trks_Sigma_hyp_exist[mask] = trks_Sigma_update_all[partition_id * n_exist_bern + bern_id]

                log_weight_hyp_exist[mask] = np.log(trks_r_ori[mask]) + log_likelihood_update_all[partition_id * n_exist_bern + bern_id]

        ####### update new bern #######
        log_weight_hyp_new = np.zeros((n_new_bern, M_new))
        trks_r_hyp_new = np.zeros((n_new_bern, M_new))

        trks_x_ref_hyp_new = np.zeros((n_new_bern, M_new, 13))
        trks_P_hyp_new = np.zeros((n_new_bern, M_new, 13, 13))
        trks_mu_hyp_new = np.zeros((n_new_bern, M_new, N_T, 9))
        trks_Sigma_hyp_new = np.zeros((n_new_bern, M_new, N_T, 9, 9))

        if n_new_bern:
            # temporary variables
            assign_new = assignments[n_exist_bern:, :M_new]

            # update miss detected tracks
            mask_miss = assign_new == -1
            trks_r_hyp_new[mask_miss] = 0
            log_weight_hyp_new[mask_miss] = 0

            # update with only radar measurements (typ=0) 
            # only keypoints measurements (typ=1)
            # both radar and keypoints measurements (typ=2)
            for typ in [1, 2]:
                mask = np.isin(assign_new, np.where(cluster_typ == typ))
                if not np.any(mask):
                    continue
                assign_id = assign_new[mask].astype(int)
                trks_r_hyp_new[mask] = 1.0

                log_weight_hyp_new[mask] = new_log_weight_update[assign_id]

                trks_x_ref_hyp_new[mask], trks_P_hyp_new[mask] = new_x_ref_update[assign_id], new_P_update[assign_id]
                trks_mu_hyp_new[mask], trks_Sigma_hyp_new[mask] = new_mu_update[assign_id], new_Sigma_update[assign_id]
            
        # stack all updated information and association results
        log_weight_hyp_all = np.concatenate((log_weight_hyp_exist, log_weight_hyp_new), axis=0)
        trks_r_hyp_all = np.concatenate((trks_r_hyp_exist, trks_r_hyp_new), axis=0)
        trks_x_ref_hyp_all = np.concatenate((trks_x_ref_hyp_exist, trks_x_ref_hyp_new), axis=0)
        trks_P_hyp_all = np.concatenate((trks_P_hyp_exist, trks_P_hyp_new), axis=0)
        trks_mu_hyp_all = np.concatenate((trks_mu_hyp_exist, trks_mu_hyp_new), axis=0)
        trks_Sigma_hyp_all = np.concatenate((trks_Sigma_hyp_exist, trks_Sigma_hyp_new), axis=0)

        ####### TPMB approximation #######
        weights_hyp = normalize_weight(log_weight_hyp_all)

        trk_r = np.sum(weights_hyp[np.newaxis, :] * trks_r_hyp_all, axis=1)
        mask_not_zero = trk_r > eps

        trk_x_ref = np.zeros((n_exist_bern + n_new_bern, 13))
        trk_P = np.zeros((n_exist_bern + n_new_bern, 13, 13))
        trk_mu = np.zeros((n_exist_bern + n_new_bern, N_T, 9))
        trk_Sigma = np.zeros((n_exist_bern + n_new_bern, N_T, 9, 9))

        trk_x_ref[mask_not_zero] = np.sum((weights_hyp[np.newaxis, :] * trks_r_hyp_all[mask_not_zero] / trk_r[mask_not_zero, np.newaxis])[..., np.newaxis] * trks_x_ref_hyp_all[mask_not_zero], axis=1)
        trk_P[mask_not_zero] = np.sum((np.square(weights_hyp[np.newaxis, :] * trks_r_hyp_all[mask_not_zero] / trk_r[mask_not_zero, np.newaxis]))[..., np.newaxis, np.newaxis] * trks_P_hyp_all[mask_not_zero], axis=1)
        trk_mu[mask_not_zero] = np.sum((weights_hyp[np.newaxis, :] * trks_r_hyp_all[mask_not_zero] / trk_r[mask_not_zero, np.newaxis])[..., np.newaxis, np.newaxis] * trks_mu_hyp_all[mask_not_zero], axis=1)
        trk_Sigma[mask_not_zero] = np.sum((weights_hyp[np.newaxis, :] * trks_r_hyp_all[mask_not_zero] / trk_r[mask_not_zero, np.newaxis])[..., np.newaxis, np.newaxis, np.newaxis] * trks_Sigma_hyp_all[mask_not_zero], axis=1)

        ####### add updated states to trks_state #######
        self.trks_r = trk_r
        if n_exist_bern > 0:
            for i, trk in enumerate(self.trks_state):
                trk.trk_x_ref[-1] = trk_x_ref[i]
                trk.trk_P[-1] = trk_P[i]
                trk.trk_mu[-1] = trk_mu[i]
                trk.trk_Sigma[-1] = trk_Sigma[i]
        time_now = np.max(ppp.trks_t)
        for i in range(n_exist_bern, trk_r.shape[0]):
            self.trks_t = np.append(self.trks_t, time_now)
            self.trks_t_id = np.append(self.trks_t_id, i-n_exist_bern)
            self.trks_state.append(State_trk(trk_x_ref[i], np.zeros(13), trk_P[i], trk_mu[i], trk_Sigma[i]))
        pass
        

    def prune(self):
        remain_mask = self.trks_r > threshold_r
        self.trks_r = self.trks_r[remain_mask]
        self.trks_t = self.trks_t[remain_mask]
        self.trks_t_id = self.trks_t_id[remain_mask]
        self.trks_state = [self.trks_state[i] for i in range(len(self.trks_state)) if remain_mask[i]]

    def recycle(self, ppp):
        remain_mask = self.trks_r > threshold_cycle
        ppp.weights = np.append(ppp.weights, self.trks_r[~remain_mask])
        ppp.trks_t = np.append(ppp.trks_t, self.trks_t[~remain_mask])
        ppp.trks_state.extend([self.trks_state[i] for i in range(len(self.trks_state)) if not remain_mask[i]])

        self.trks_r = self.trks_r[remain_mask]
        self.trks_t = self.trks_t[remain_mask]
        self.trks_t_id = self.trks_t_id[remain_mask]
        self.trks_state = [self.trks_state[i] for i in range(len(self.trks_state)) if remain_mask[i]]

    def merge(self):
        return


class PPP:
    def __init__(self):
        self.weights = np.array([])
        self.trks_t = np.array([])
        self.trks_state = []

    def extend_ppp(self, ppp_):
        self.weights = np.append(self.weights, ppp_.weights)
        self.trks_t = np.append(self.trks_t, ppp_.trks_t)
        self.trks_state.extend(copy.deepcopy(ppp_.trks_state))

    def predict(self, birth):
        self.weights *= Ps

        if len(self.trks_state) > 0:
            # stack all the state that need to be predicted
            trks_x_ref_con = np.stack(
                [trk.trk_x_ref[-1] for trk in self.trks_state], axis=0)
            trks_P_con = np.stack(
                [trk.trk_P[-1] for trk in self.trks_state], axis=0)
            trks_mu_con = np.stack(
                [trk.trk_mu[-1] for trk in self.trks_state], axis=0)
            trks_Sigma_con = np.stack(
                [trk.trk_Sigma[-1] for trk in self.trks_state], axis=0)

            # predict next state of every track
            trks_x_ref_con, trks_P_con, trks_mu_con, trks_Sigma_con = predict_batch(
                trks_x_ref_con, trks_P_con, trks_mu_con, trks_Sigma_con)

            # add predictions on trks_state
            for i, trk in enumerate(self.trks_state):
                trk.trk_x_ref = np.vstack((trk.trk_x_ref, trks_x_ref_con[i:i+1]))
                trk.dx = np.vstack((trk.trk_dx, np.zeros((1, 13))))
                trk.trk_P = np.vstack((trk.trk_P, trks_P_con[i:i+1]))
                trk.trk_mu = np.vstack((trk.trk_mu, trks_mu_con[i:i+1]))
                trk.trk_Sigma = np.vstack((trk.trk_Sigma, trks_Sigma_con[i:i+1]))

        assert isinstance(birth, PPP), 'birth must be class PPP'
        self.extend_ppp(birth)
    
    def update(self):
        self.weights *= q_d_r * q_d_c

    def prune(self):
        remain_mask = self.weights > threshold_u
        self.weights = self.weights[remain_mask]
        self.trks_t = self.trks_t[remain_mask]
        self.trks_state = [self.trks_state[i] for i in range(len(self.trks_state)) if remain_mask[i]]

    def merge(self):
        return


def TPMB_RCfusion_predict(ppp, mb, birth):
    ppp.predict(birth)
    mb.predict()
    return ppp, mb


def TPMB_RCfusion_update(ppp, mb, pcl, kps, pose_shape, mdn, s2p):
    mb.update(ppp, pcl, kps, pose_shape, mdn, s2p)
    ppp.update()
    return ppp, mb

def TPMB_RCfusion_prune(ppp, mb):
    ppp.prune()
    mb.prune()
    mb.recycle(ppp)
    return ppp, mb

def TPMB_RCfusion_merge(ppp, mb):
    ppp.merge()
    mb.merge()
    return ppp, mb


def predict_batch(x_ref, P, mu, Sigma):
    assert x_ref.ndim == 2, 'x_ref dim error'
    assert P.ndim == 3, 'P dim error'
    assert mu.ndim == 3, 'mu dim error'
    assert Sigma.ndim == 4, 'Sigma dim error'

    # get states
    p_ref = x_ref[:, :3]
    v_ref = x_ref[:, 3]
    theta_ref = x_ref[:, 4:7]
    omega_ref = x_ref[:, 7:10]
    ext_ref = x_ref[:, 10:13]

    # convert rotation vector to rotation matrix
    R_ref = rotation_vectors_to_matrices(theta_ref)

    # predict x_ref
    ret_x_ref = f(p_ref, v_ref, R_ref, omega_ref, ext_ref)

    # get phi matrix and predict P
    Phi = get_phi(v_ref, omega_ref, R_ref)
    ret_P = Phi @ P @ Phi.transpose(0, 2, 1) + W * dt

    # get Phi_vartheta
    Phi_vartheta = get_phi_vartheta()
    ret_mu = (Phi_vartheta @ mu[:, :, :, np.newaxis]).squeeze(axis=-1)
    ret_Sigma = Phi_vartheta @ Sigma @ Phi_vartheta.T + W_vartheta * dt

    return ret_x_ref, ret_P, ret_mu, ret_Sigma

def f(p_ref, v_ref, R_ref, omega_ref, ext_ref):
    p_ref = np.atleast_2d(p_ref)
    v_ref = np.atleast_1d(v_ref)
    R_ref = np.expand_dims(R_ref, axis=0) if R_ref.ndim == 2 else R_ref
    omega_ref = np.atleast_2d(omega_ref)
    ext_ref = np.atleast_2d(ext_ref)

    n = len(v_ref)

    norm_omega_ref = np.linalg.norm(omega_ref, axis=1)
    norm_omega_ref_reshape = norm_omega_ref[:, np.newaxis, np.newaxis]
    mask = norm_omega_ref > eps

    n_mask = np.count_nonzero(mask)
    trans_mat = np.tile(np.eye(3), (n, 1, 1)) * dt
    # If all norms are smaller than eps, return identity matrices
    if n_mask > 0:
        skew = skew_symmetry(omega_ref[mask] / norm_omega_ref[mask, np.newaxis])
        trans_mat[mask] += 1 / norm_omega_ref_reshape[mask] * skew * (1 - np.cos(norm_omega_ref_reshape[mask] * dt)) + skew @ skew * (dt - np.sin(norm_omega_ref_reshape[mask] * dt) / norm_omega_ref_reshape[mask])
    
    ret_p_ref = p_ref + (v_ref[:, np.newaxis, np.newaxis] * trans_mat @ R_ref @ u_d[:, np.newaxis]).squeeze()
    ret_v_ref = v_ref

    ret_R_ref = exp_skew_v(omega_ref * dt) @ R_ref
    ret_theta_ref = rotation_matrics_to_vectors(ret_R_ref)

    ret_omega_ref = omega_ref
    ret_ext_ref = ext_ref

    return np.hstack((ret_p_ref, ret_v_ref[:, np.newaxis], ret_theta_ref, ret_omega_ref, ret_ext_ref))


def partition_based_on_projection(pcl, kps, corner_kps, pose_shape, bottom_middle_pos, bottom_middle_P, s2p):
    n_kps = len(corner_kps)

    corner_kps = corner_kps.reshape([-1, 2])

    # first: cluster point cloud using DBSCAN
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    clus_pcl_dbscan = dbscan.fit_predict(pcl)

    # divide pcl to pcl_clusters
    pcl_clusters = []
    for label in np.unique(clus_pcl_dbscan[clus_pcl_dbscan!=-1]):
        cloud = pcl[clus_pcl_dbscan == label]
        pcl_clusters.append(cloud)
    n_pcl_clus = len(pcl_clusters)
    
    # stack all pcl clusters
    all_pcl = np.vstack(pcl_clusters)

    # project pcl to image
    pcl_pix = [(cv2.projectPoints(cloud, np.zeros(3), np.zeros(3), s2p.cameraMatrix, s2p.distCoeffs)[0]).squeeze(axis=1) for cloud in pcl_clusters]
    
    # split kps id and pixel value
    kps_pix = [np.asarray(kp)[np.asarray(kp)[:, 0] < 24, 1:] for kp in kps]
    kps_pix_id = [(np.asarray(kp)[np.asarray(kp)[:, 0] < 24, 0]).astype(int) for kp in kps]
    n_kps_clus = len(kps_pix)

    # calculate mean and covariance of kps clusters
    kps_cov = np.array([np.cov(pix, rowvar=False) for pix in kps_pix])
    kps_mean = np.array([np.mean(pix, axis=0) for pix in kps_pix])
    
    kps_cov = kps_cov.reshape([-1, 2, 2])
    kps_mean = kps_mean.reshape([-1, 2])

    # calculate mahalanobis distance between projected pcl and kps pixel
    pcl_clus_kps_mahalanobis = [np.squeeze((p[np.newaxis, :, np.newaxis, :]-kps_mean[:, np.newaxis, np.newaxis, :]) @ np.linalg.inv(kps_cov)[:, np.newaxis, :, :] @ (p[np.newaxis, :, :, np.newaxis]-kps_mean[:, np.newaxis, :, np.newaxis]), axis=(-2, -1)) for p in pcl_pix]
    
    if n_kps == 0:
        pcl_clusters_new, pcl_clusters_new_kps_mahalanobis = pcl_clusters, np.zeros((n_pcl_clus, n_kps))
        n_pcl_clusters_new = n_pcl_clus
    else:
        #### Ablation 2 ####
        # split DBSCAN clusters based on kps
        # calculate new pcl_clusters and mahalanobis distance 
        # between new pcl clusters and kps pixel
        ## with doped clustering
        pcl_clusters_new = []
        pcl_clusters_new_kps_mahalanobis = []

        for i, pcl_ass in enumerate(pcl_clus_kps_mahalanobis):
            asso_kps_id = np.argmin(pcl_ass, axis=0)
            asso_kps_dis = np.min(pcl_ass, axis=0)

            asso_kps = np.unique(asso_kps_id)
            # associated with different kps
            for id_kps in asso_kps:
                if np.count_nonzero(asso_kps_id == id_kps) < dbscan_min_samples:
                    continue
                now_dis = np.ones(n_kps_clus) * 1e3
                pcl_clusters_new.append(pcl_clusters[i][asso_kps_id == id_kps])
                now_dis[id_kps] = np.mean(asso_kps_dis[asso_kps_id == id_kps])
                pcl_clusters_new_kps_mahalanobis.append(now_dis)

        pcl_clusters_new_kps_mahalanobis = np.asarray(pcl_clusters_new_kps_mahalanobis)
        n_pcl_clusters_new = len(pcl_clusters_new)

    ## without doped clustering
    # pcl_clusters_new, pcl_clusters_new_kps_mahalanobis = pcl_clusters, np.array([np.mean(item, axis=1) for item in pcl_clus_kps_mahalanobis])
    # n_pcl_clusters_new = n_pcl_clus

    # association and split measurements based on the distance matrix
    Z_p_c = []
    s_i = []
    Z_p_cor_pix = []
    Z_p_pose = []
    Z_p_shape = []
    Z_p_skeleton = []
    Z_p_middle_pos = []
    Z_p_middle_P = []
    valid_r = []
    Z_p_r = []
    clus_typ = []

    vis_c = np.zeros(n_kps_clus)
    vis_r = np.zeros(n_pcl_clusters_new)

    cost = pcl_clusters_new_kps_mahalanobis
    cost[cost >= gate_2d] = 1e3
    if cost.shape[0] > cost.shape[1]:
        cost = cost.T
        # row is kps, col is pcl
        if len(cost) != 0:
            mgen = Murty(cost)
            ok, _, sol = mgen.draw()
            if ok:
                for i in range(cost.shape[0]):
                    now = np.zeros((N_T, 2))
                    now_s_i = np.zeros(N_T).astype(bool)
                    now_r = pcl_clusters_new[sol[i]]
                    if cost[i][sol[i]] < gate_2d:
                        now[kps_pix_id[i]] = kps_pix[i]
                        Z_p_c.append(now)
                        now_s_i[kps_pix_id[i]] = True
                        s_i.append(now_s_i)

                        Z_p_cor_pix.append(corner_kps[i*N_c: (i+1)*N_c, :])

                        Z_p_pose.append(pose_shape['rvecs'][i])

                        Z_p_shape.append(pose_shape['sizes'][i])

                        Z_p_skeleton.append(pose_shape['skeletons'][i])

                        Z_p_middle_pos.append(bottom_middle_pos[i])

                        Z_p_middle_P.append(bottom_middle_P[i])

                        Z_p_r.append(now_r)

                        vis_c[i] = 1
                        vis_r[sol[i]] = 1
                        clus_typ.append(2)
                    # else:
                    #     # not associated with kps (clutter)
                    #     vis_r[sol[i]] = 1
                for i in range(len(kps_pix)):
                    if not vis_c[i]:
                        now = np.zeros((N_T, 2))
                        now_s_i = np.zeros(N_T).astype(bool)
                        now[kps_pix_id[i]] = kps_pix[i]
                        Z_p_c.append(now)
                        now_s_i[kps_pix_id[i]] = True

                        Z_p_cor_pix.append(corner_kps[i*N_c: (i+1)*N_c, :])

                        Z_p_pose.append(pose_shape['rvecs'][i])

                        Z_p_shape.append(pose_shape['sizes'][i])

                        Z_p_skeleton.append(pose_shape['skeletons'][i])

                        Z_p_middle_pos.append(bottom_middle_pos[i])

                        Z_p_middle_P.append(bottom_middle_P[i])

                        s_i.append(now_s_i)
                        Z_p_r.append([])
                        clus_typ.append(1)

                for i in range(len(pcl_clusters_new)):
                    if not vis_r[i]:
                        now = np.zeros((N_T, 2))
                        now_s_i = np.zeros(N_T).astype(bool)
                        now_r = pcl_clusters_new[i]

                        Z_p_c.append(now)
                        s_i.append(now_s_i)

                        Z_p_cor_pix.append(np.zeros((N_c, 2)))

                        Z_p_pose.append(np.zeros(3))

                        Z_p_shape.append(np.zeros(3))

                        Z_p_skeleton.append(np.zeros((24, 3)))

                        Z_p_middle_pos.append(np.zeros(3))

                        Z_p_middle_P.append(np.eye(3))

                        Z_p_r.append(now_r)
                        clus_typ.append(0)
            else:   
                raise Exception("No partition solution!")
        else:
            for i in range(len(pcl_clusters_new)):
                if not vis_r[i]:
                    now = np.zeros((N_T, 2))
                    now_s_i = np.zeros(N_T).astype(bool)
                    now_r = pcl_clusters_new[i]

                    Z_p_c.append(now)
                    s_i.append(now_s_i)

                    Z_p_cor_pix.append(np.zeros((N_c, 2)))

                    Z_p_pose.append(np.zeros(3))

                    Z_p_shape.append(np.zeros(3))

                    Z_p_skeleton.append(np.zeros((24, 3)))

                    Z_p_middle_pos.append(np.zeros(3))

                    Z_p_middle_P.append(np.eye(3))

                    Z_p_r.append(now_r)
                    clus_typ.append(0)
    else:
        # row is pcl, col is kps
        if len(cost) != 0:
            mgen = Murty(cost)
            ok, _, sol = mgen.draw()
            if ok:
                for i in range(cost.shape[0]):
                    now = np.zeros((N_T, 2))
                    now_s_i = np.zeros(N_T).astype(bool)
                    now_r = pcl_clusters_new[i]
                    if cost[i][sol[i]] < gate_2d:
                        now[kps_pix_id[sol[i]]] = kps_pix[sol[i]]
                        Z_p_c.append(now)
                        now_s_i[kps_pix_id[sol[i]]] = True
                        s_i.append(now_s_i)

                        Z_p_cor_pix.append(corner_kps[sol[i]*N_c: (sol[i]+1)*N_c, :])

                        Z_p_pose.append(pose_shape['rvecs'][sol[i]])

                        Z_p_shape.append(pose_shape['sizes'][sol[i]])

                        Z_p_skeleton.append(pose_shape['skeletons'][sol[i]])

                        Z_p_middle_pos.append(bottom_middle_pos[sol[i]])

                        Z_p_middle_P.append(bottom_middle_P[sol[i]])

                        Z_p_r.append(now_r)

                        vis_c[sol[i]] = 1
                        vis_r[i] = 1
                        clus_typ.append(2)
                    # else:
                    #     # not associated with kps (clutter)
                    #     vis_r[i] = 1
                for i in range(len(kps_pix)):
                    if not vis_c[i]:
                        now = np.zeros((N_T, 2))
                        now_s_i = np.zeros(N_T).astype(bool)
                        now[kps_pix_id[i]] = kps_pix[i]
                        Z_p_c.append(now)
                        now_s_i[kps_pix_id[i]] = True
                        s_i.append(now_s_i)

                        Z_p_cor_pix.append(corner_kps[i*N_c: (i+1)*N_c, :])

                        Z_p_pose.append(pose_shape['rvecs'][i])

                        Z_p_shape.append(pose_shape['sizes'][i])

                        Z_p_skeleton.append(pose_shape['skeletons'][i])

                        Z_p_middle_pos.append(bottom_middle_pos[i])

                        Z_p_middle_P.append(bottom_middle_P[i])

                        Z_p_r.append([])
                        clus_typ.append(1)

                for i in range(len(pcl_clusters_new)):
                    if not vis_r[i]:
                        now = np.zeros((N_T, 2))
                        now_s_i = np.zeros(N_T).astype(bool)
                        now_r = pcl_clusters_new[i]

                        Z_p_c.append(now)
                        s_i.append(now_s_i)

                        Z_p_cor_pix.append(np.zeros((N_c, 2)))

                        Z_p_pose.append(np.zeros(3))

                        Z_p_shape.append(np.zeros(3))

                        Z_p_skeleton.append(np.zeros((24, 3)))

                        Z_p_middle_pos.append(np.zeros(3))

                        Z_p_middle_P.append(np.eye(3))

                        Z_p_r.append(now_r)
                        clus_typ.append(0)
            else:
                raise Exception("No partition solution!")
        else:
            for i in range(len(kps_pix)):
                if not vis_c[i]:
                    now = np.zeros((N_T, 2))
                    now_s_i = np.zeros(N_T).astype(bool)
                    now[kps_pix_id[i]] = kps_pix[i]
                    Z_p_c.append(now)
                    now_s_i[kps_pix_id[i]] = True
                    s_i.append(now_s_i)

                    Z_p_cor_pix.append(corner_kps[i*N_c: (i+1)*N_c, :])

                    Z_p_pose.append(pose_shape['rvecs'][i])

                    Z_p_shape.append(pose_shape['sizes'][i])

                    Z_p_skeleton.append(pose_shape['skeletons'][i])

                    Z_p_middle_pos.append(bottom_middle_pos[i])

                    Z_p_middle_P.append(bottom_middle_P[i])

                    Z_p_r.append([])
                    clus_typ.append(1)

    Z_p_c = np.asarray(Z_p_c)
    s_i = np.asarray(s_i)
    Z_p_cor_pix = np.asarray(Z_p_cor_pix)

    Z_p_pose = np.asarray(Z_p_pose)
    Z_p_shape = np.asarray(Z_p_shape)
    Z_p_skeleton = np.asarray(Z_p_skeleton)

    Z_p_middle_pos = np.asarray(Z_p_middle_pos)
    Z_p_middle_P = np.asarray(Z_p_middle_P)

    clus_typ = np.asarray(clus_typ)

    n_new_bern = len(Z_p_c)
    len_p_r = [len(p) for p in Z_p_r]
    if len(len_p_r) > 0:
        max_len_r = np.max(len_p_r)
    else:
        max_len_r = 0
    valid_r = np.array([np.append(np.ones(len_p_r[i]), np.zeros(max_len_r-len_p_r[i])) for i in range(n_new_bern)]).astype(bool)
    Z_p_r = np.array([
        np.pad(p, ((0, max_len_r - len(p)), (0, 0)), constant_values=0) if len(p) > 0 else np.zeros((max_len_r, 3))
        for p in Z_p_r
    ])
    return Z_p_c, s_i, Z_p_cor_pix, Z_p_pose, Z_p_shape, Z_p_skeleton, Z_p_middle_pos, Z_p_middle_P, valid_r, Z_p_r, clus_typ

def update_using_measurements(
    trks_x_ref, trks_dx, trks_P, 
    trks_mu, trks_Sigma, 
    clus_id, Z_p_r, valid_r, 
    Z_p_c, s_i, 
    Z_p_cor_pix, Z_p_pose, Z_p_shape, Z_p_skeleton,
    Z_p_middle_pos, Z_p_middle_P, 
    typ, s2p):

    if len(trks_x_ref) == 0:
        return [np.array([])] * 5
    # reshape variables
    trks_x_ref = trks_x_ref if trks_x_ref.ndim == 2 else np.expand_dims(
        trks_x_ref, axis=0)
    trks_dx = trks_dx if trks_dx.ndim == 2 else np.expand_dims(trks_dx, axis=0)
    trks_P = trks_P if trks_P.ndim == 3 else np.expand_dims(trks_P, axis=0)
    trks_mu = trks_mu if trks_mu.ndim == 3 else np.expand_dims(trks_mu, axis=0)
    trks_Sigma = trks_Sigma if trks_Sigma.ndim == 4 else np.expand_dims(
        trks_Sigma, axis=0)

    ret_x_ref = trks_x_ref.copy()
    ret_P = trks_P.copy()
    ret_mu = trks_mu.copy()
    ret_Sigma = trks_Sigma.copy()

    n_obj = trks_x_ref.shape[0]

    log_likeli = np.zeros(n_obj)
    likeli_thres = np.zeros(n_obj)

    # VB approximation
    for it in range(N_iter):
        p_ref = ret_x_ref[:, :3]
        theta = ret_x_ref[:, 4:7]
        omega = ret_x_ref[:, 7:10]
        ext = ret_x_ref[:, 10:13]
        R = rotation_vectors_to_matrices(theta)
        R_line = Star(Sqrt(Q_inv)).T @ R
        J = get_J(theta)
        P = ret_P
        P_inv = np.linalg.inv(P)
        Cov_p = P[:, :3, :3]
        Cov_theta = P[:, 4:7, 4:7]
        Cov_theta_p = P[:, 4:7, 0:3]

        mu = ret_mu
        E_u = mu[..., :3]
        Sigma = ret_Sigma
        Sigma_inv = np.linalg.inv(Sigma)
        Cov_u = Sigma[..., :3, :3]
        Cov_u_inv = np.linalg.inv(Cov_u)

        # Pseudo measurement
        varpi_sym = mu[..., 3:6]
        
        H_xs = np.zeros_like(P)
        h_xs = np.zeros_like(ret_x_ref)
        H_varthetas = np.zeros_like(Sigma)
        h_varthetas = np.zeros_like(mu)

        if typ == 0 or typ == 2:
            ### has radar measurements ###
            z_r = Z_p_r[clus_id][:, np.newaxis, :, :]

            # Calculate parameters for A (association of pcl and reflection points)
            E_log_pi = 0.1 * ((-p_ref[:, np.newaxis, :] @ (R @ E_u.transpose(0, 2, 1))) / (np.linalg.norm(p_ref, axis=-1, keepdims=True) * np.linalg.norm(E_u, axis=-1))[:, np.newaxis, :] - 1)

            E_zeta = p_ref[:, np.newaxis, :] + (R @ E_u.transpose(0, 2, 1)).transpose(0, 2, 1)
            E_zeta_Q = (Q_inv @ Cov_p).transpose(1, 2, 0).trace()[:, np.newaxis] + (R.transpose(0, 2, 1)[:, np.newaxis, :, :] @ Q_inv @ R[:, np.newaxis, :, :] @ Cov_u + (R_line.transpose(0, 2, 1) @ diag_n(J @ Cov_theta @ J.transpose(0, 2, 1), 3) @ R_line)[:, np.newaxis, :, :] @ (Cov_u + E_u[:, :, :, np.newaxis] @ E_u[:, :, np.newaxis, :]) - 2 * Q_inv @ skew_symmetry((R @ E_u.transpose(0, 2, 1)).transpose(0, 2, 1)) @ J[:, np.newaxis, :, :] @ Cov_theta_p[:, np.newaxis, :, :]).transpose((2, 3, 0, 1)).trace()

            diff = z_r - E_zeta[:, :, np.newaxis, :]
            E_z_zeta_Q = np.squeeze(diff[:, :, :, np.newaxis, :] @ Q_inv @ diff[:, :, :, :, np.newaxis], axis=(-1, -2)) + E_zeta_Q[:, :, np.newaxis]
            sub = E_log_pi - 1 / 2 * E_z_zeta_Q.transpose(0, 2, 1)

            # sub = np.tile(E_log_pi, [1, z_r.shape[2], 1])

            sub_t = sub - np.max(sub, axis=-1, keepdims=True)
            upsilon = np.exp(sub_t) / np.sum(np.exp(sub_t), axis=-1, keepdims=True)

            # calculate E_A
            valid_n_t = np.tile(valid_r[clus_id], (N_T, 1, 1)).transpose(1, 2, 0)
            n_t = upsilon.sum(axis = 1, where=valid_n_t)
            n_t[n_t < eps] = eps

            valid_r_id = np.tile(valid_r[clus_id], (N_T, 3, 1, 1)).transpose(2, 0, 3, 1)
            z_line = np.sum(upsilon.transpose(0, 2, 1)[:, :, :, np.newaxis] * z_r, axis=2, where=valid_r_id) / n_t[:, :, np.newaxis]
            valid_r_id_Z = np.tile(valid_r[clus_id], (N_T, 3, 3, 1, 1)).transpose(3, 0, 4, 1, 2)
            diff = z_r - z_line[:, :, np.newaxis, :]
            Z_line = np.sum(upsilon.transpose(0, 2, 1)[:, :, :, np.newaxis, np.newaxis] * diff[:, :, :, :, np.newaxis] @ diff[:, :, :, np.newaxis, :], axis=2, where=valid_r_id_Z)

            ### Calculate parameters for x ###
            # part 1
            Q_n_inv = n_t[:, :, np.newaxis, np.newaxis] * Q_inv
            J_tilde = Star(Sqrt(Q_n_inv)).transpose(
                0, 1, 3, 2) @ J[:, np.newaxis, :, :]
            H_1_x = H_r
            h_1_x = z_line - (R @ E_u.transpose(0, 2, 1)).transpose(0, 2, 1) - p_ref[:, np.newaxis, :]

            # add to Hs, hs
            H_xs += np.sum(H_1_x.T @ Q_n_inv @ H_1_x, axis=1)
            h_xs += np.sum(H_1_x.T @ Q_n_inv @ h_1_x[:, :, :, np.newaxis], axis=1).squeeze(axis=-1)

            if it == 0:
                diff = h_1_x
                Q_n = np.linalg.inv(Q_n_inv)
                Q_pcl = (H_1_x @ trks_P @ H_1_x.T)[:, np.newaxis, :, :] + Q_n

                det_Q_pcl = np.linalg.det(Q_pcl)
                log_likeli += np.sum(-3/2*np.log(2*np.pi) - 1/2*np.log(det_Q_pcl) - 1/2*np.squeeze(diff[:, :, np.newaxis, :] @ np.linalg.inv(Q_pcl) @ diff[:, :, :, np.newaxis], axis=(-1, -2)), axis=-1)
                likeli_thres += np.sum((1/2*gate_3d + 3/2*np.log(2*np.pi) + 1/2*np.log(det_Q_pcl)), axis=-1)

            ### Calculate parameters for vartheta ###
            # part 1
            R_tilde = Star(Sqrt(Q_n_inv)).transpose((0, 1, 3, 2)) @ R[:, np.newaxis, :, :]
            H_1_vartheta = (R @ H_u)[:, np.newaxis, :, :]
            h_1_vartheta = z_line - p_ref[:, np.newaxis, :]

            # add to H_varthetas, h_varthetas
            H_varthetas += H_1_vartheta.transpose((0, 1, 3, 2)) @ Q_n_inv @ H_1_vartheta
            h_varthetas += (H_1_vartheta.transpose((0, 1, 3, 2)) @ Q_n_inv @ h_1_vartheta[:, :, :, np.newaxis]).squeeze(axis=-1)

            # part 2
            Q_vartheta_inv = R_tilde.transpose(0, 1, 3, 2) @ diag_n(J @ Cov_theta @ J.transpose(0, 2, 1), 3)[:, np.newaxis, :, :] @ R_tilde
            Q_vartheta = np.linalg.inv(Q_vartheta_inv)
            h_2_vartheta = (-Q_vartheta.transpose(0, 1, 3, 2) @ R[:, np.newaxis, :, :].transpose(0, 1, 3, 2) @ Star(Q_n_inv) @ Pentacle(J @ Cov_theta_p)[:, np.newaxis, :, np.newaxis]).squeeze(axis=-1)

            # add to H_varthetas, h_varthetas
            H_varthetas += H_u.T @ Q_vartheta_inv @ H_u
            h_varthetas += (H_u.T @ Q_vartheta_inv @ h_2_vartheta[:, :, :, np.newaxis]).squeeze(axis=-1)

        if typ == 1 or typ == 2:
            ### has pixel keypoints measurements ###
            ### Calculate parameters for x ###
            # part 1 (update position)
            H_2_x = H_r
            h_2_x = Z_p_middle_pos[clus_id] - p_ref
            Q_2_x = Z_p_middle_P[clus_id]
            Q_2_x_inv = np.linalg.inv(Q_2_x)
            # add to Hs, hs
            H_xs += H_2_x.T @ Q_2_x_inv @ H_2_x
            h_xs += (H_2_x.T @ Q_2_x_inv @ h_2_x[:, :, np.newaxis]).squeeze(axis=-1)

            # part 2 (update width, length and height)
            H_3_x = H_ext
            h_3_x = Z_p_shape[clus_id] - ext
            # add to Hs, hs
            H_xs += H_3_x.T @ Q_model_inv @ H_3_x
            h_xs += (H_3_x.T @ Q_model_inv @ h_3_x[:, :, np.newaxis]).squeeze(axis=-1)

            H_4_x = H_theta
            h_4_x = Z_p_pose[clus_id] - theta
            # add to Hs, hs
            H_xs += H_4_x.T @ Q_pose_inv @ H_4_x
            h_xs += (H_4_x.T @ Q_pose_inv @ h_4_x[:, :, np.newaxis]).squeeze(axis=-1)

            ### Calculate parameters for vartheta ###
            H_3_vartheta = H_varpi
            h_3_vartheta = Z_p_skeleton[clus_id]

            # add to H_varthetas, h_varthetas
            H_varthetas += H_3_vartheta.T @ Q_model_inv @ H_3_vartheta
            h_varthetas += (H_3_vartheta.T @ Q_model_inv @ h_3_vartheta[:, :, :, np.newaxis]).squeeze(axis=-1)

            if it == 0:
                # position measurements
                diff = h_2_x
                Q_pos = H_2_x @ trks_P @ H_2_x.T + Q_2_x
                det_Q_pos = np.linalg.det(Q_pos)
                log_likeli += -3/2*np.log(2*np.pi) - 1/2*np.log(det_Q_pos) - 1/2*np.squeeze(diff[:, np.newaxis, :] @ np.linalg.inv(Q_pos) @ diff[:, :, np.newaxis], axis=(-1, -2))
                likeli_thres += 1/2*gate_3d + 3/2*np.log(2*np.pi) + 1/2*np.log(det_Q_pos)

                # pixel measurements
                varpi_S = p_ref[:, np.newaxis, :]
                varpi_I = s2p.cameraMatrix @ varpi_S.transpose(0, 2, 1)
                varpi_I /= varpi_I[:, 2:3, :]
                varpi_I = varpi_I[:, :2, :].transpose(0, 2, 1)

                f_dx, f_dy = s2p.cameraMatrix[0, 0], s2p.cameraMatrix[1, 1]
                u_0, v_0 = s2p.cameraMatrix[0, 2], s2p.cameraMatrix[1, 2]

                p_varpi_I_p_varpi_S = np.zeros([n_obj, 1,  2, 3])
                p_varpi_I_p_varpi_S[..., 0, 0] = f_dx / varpi_S[..., 2]
                p_varpi_I_p_varpi_S[..., 0, 2] = -varpi_S[..., 0] * f_dx / varpi_S[..., 2]**2
                p_varpi_I_p_varpi_S[..., 1, 1] = f_dy / varpi_S[..., 2]
                p_varpi_I_p_varpi_S[..., 1, 2] = -varpi_S[..., 1] * f_dy / varpi_S[..., 2]**2

                diff = np.mean(Z_p_cor_pix[clus_id, :4, :] - varpi_I + (p_varpi_I_p_varpi_S @ varpi_S[..., np.newaxis]).squeeze(axis=-1), axis=1)
                Q_V_c = np.array([np.cov(pix[valid], rowvar=False) for pix, valid in zip(Z_p_c[clus_id], s_i[clus_id])]) / 10
                Q_kps = Q_V_c + (p_varpi_I_p_varpi_S @ Cov_p[:, np.newaxis, :, :] @ p_varpi_I_p_varpi_S.transpose(0, 1, 3, 2)).squeeze(axis=1)

                det_Q_kps = np.linalg.det(Q_kps)
                log_likeli += -3/2*np.log(2*np.pi) - 1/2*np.log(det_Q_kps) - 1/2*np.squeeze(diff[:, np.newaxis, :] @ np.linalg.inv(Q_kps) @ diff[:, :, np.newaxis], axis=(-1, -2))
                likeli_thres += 1/2*gate_3d + 3/2*np.log(2*np.pi) + 1/2*np.log(det_Q_kps)


        ### pseudo measurement ###
        # H_5_x = road_vector[np.newaxis, :] @ H_r
        # h_5_x = -(road_vector[np.newaxis, :] @ p_ref[:, :, np.newaxis] + d_ground)

        # # add to Hs, hs
        # H_xs += H_5_x.T @ Q_ground_inv @ H_5_x
        # h_xs += (H_5_x.T @ Q_ground_inv @ h_5_x).squeeze(axis=-1)

        # # (nearly not rotating)
        # H_6_x = ((R @ u_d[:, np.newaxis]).transpose(0, 2, 1) @ H_omega - omega[:, np.newaxis, :] @ skew_symmetry((R @ u_d[:, np.newaxis]).squeeze(axis=2)) @ J @ H_theta).reshape([-1, 1, 13])
        # h_6_x = -(R @ u_d[:, np.newaxis]).transpose(0, 2, 1) @ omega[:, :, np.newaxis]

        # # add to Hs, hs
        # H_xs += H_6_x.transpose((0, 2, 1)) @ Q_rot_inv @ H_6_x
        # h_xs += (H_6_x.transpose((0, 2, 1)) @ Q_rot_inv @ h_6_x).squeeze(axis=-1)

        ### update parameters ###
        # update parameters of x
        ret_P = np.linalg.inv(P_inv + H_xs)
        dx_new = (ret_P @ h_xs[..., np.newaxis]).squeeze(axis=-1)

        ret_x_ref += dx_new
        dx_new = np.zeros_like(dx_new)

        # update parameters of vartheta
        ret_Sigma = np.linalg.inv(Sigma_inv + H_varthetas)
        ret_mu = (ret_Sigma @ ((Sigma_inv @ mu[..., np.newaxis]).squeeze(axis=-1) + h_varthetas)[..., np.newaxis]).squeeze(axis=-1)

        if it == 0:
            if typ == 0:
                log_likeli += np.log(q_d_c)
            elif typ == 1:
                log_likeli += np.log(q_d_r)

    
    return ret_x_ref, ret_P, ret_mu, ret_Sigma, log_likeli, likeli_thres


def get_phi(v_ref, omega_ref, R_ref):
    v_ref = np.atleast_1d(v_ref)
    omega_ref = np.atleast_2d(omega_ref)
    R_ref = np.expand_dims(R_ref, axis=0) if R_ref.ndim == 2 else R_ref

    n = len(v_ref)

    M_0 = (R_ref @ u_d).reshape((n, 3, 1))
    M_1 = -v_ref[:, np.newaxis, np.newaxis] * skew_symmetry(R_ref @ u_d)
    M_2 = skew_symmetry(omega_ref)

    norm_omega_ref = np.linalg.norm(omega_ref, axis=1)
    norm_omega_ref_reshape = norm_omega_ref[:, np.newaxis, np.newaxis]
    mask = norm_omega_ref > eps

    eye_1_broadcasted = np.broadcast_to(np.eye(1), (n, 1, 1))
    eye_2_broadcasted = np.broadcast_to(np.eye(2), (n, 2, 2))
    eye_3_broadcasted = np.broadcast_to(np.eye(3), (n, 3, 3))

    S_0 = rotation_vectors_to_matrices(omega_ref * dt)
    S_1 = eye_3_broadcasted * dt
    S_2 = 1 / 2 * eye_3_broadcasted * dt**2

    S_1[mask] = S_1[mask] - \
        (M_2[mask] / norm_omega_ref_reshape[mask]**2) @ (S_0[mask] - np.eye(3) - M_2[mask] * dt)
    S_2[mask] = S_2[mask] - 1 / norm_omega_ref_reshape[mask]**2 * \
        (S_0[mask] - np.eye(3) - M_2[mask] * dt - 1 / 2 * M_2[mask] @ M_2[mask] * dt**2)

    zeros_1_2 = np.zeros((n, 1, 2))
    zeros_1_3 = np.zeros((n, 1, 3))
    zeros_2_1 = np.zeros((n, 2, 1))
    zeros_2_3 = np.zeros((n, 2, 3))
    zeros_3_1 = np.zeros((n, 3, 1))
    zeros_3_2 = np.zeros((n, 3, 2))
    zeros_3_3 = np.zeros((n, 3, 3))

    Phi = np.block([[eye_3_broadcasted,           M_0 * dt,  M_1 @ S_1,          M_1 @ S_2,          zeros_3_3],
                    [        zeros_1_3,  eye_1_broadcasted,  zeros_1_3,          zeros_1_3,          zeros_1_3],
                    [        zeros_3_3,          zeros_3_1,        S_0,                S_1,          zeros_3_3],
                    [        zeros_3_3,          zeros_3_1,  zeros_3_3,  eye_3_broadcasted,          zeros_3_3],
                    [        zeros_3_3,          zeros_3_1,  zeros_3_3,          zeros_3_3,  eye_3_broadcasted]])
    return Phi

def get_phi_vartheta():
    now = -4 * epsilon + rho ** 2
    if now < -eps:
        # overdamped
        S_3 = -1 + np.exp(-rho / 2 * dt) * np.cos(1 / 2 * np.sqrt(-now) * dt) + np.exp(-rho / 2 * dt) * rho * np.sin(1 / 2 * np.sqrt(-now) * dt) / np.sqrt(-now)
        S_4 = 2 * np.exp(-rho / 2 * dt) * np.sin(1 / 2 * np.sqrt(-now) * dt) / np.sqrt(-now)
    elif now > eps:
        # underdamped
        S_3 = -1 + np.exp(-rho / 2 * dt) * np.cosh(1 / 2 * np.sqrt(now) * dt) + np.exp(-rho / 2 * dt) * rho * np.sinh(1 / 2 * np.sqrt(now) * dt) / np.sqrt(now)
        S_4 = 2 * np.exp(-rho / 2 * dt) * np.sinh(1 / 2 * np.sqrt(now) * dt) / np.sqrt(now)
    else:
        # damped
        S_3 = -1 + np.exp(-rho / 2 * dt) + 1 / 2 * np.exp(-rho / 2 * dt) * rho * dt
        S_4 = np.exp(-rho / 2 * dt) * dt

    concat = np.zeros((3, 3))
    concat[0, 0] = 1
    concat[1, 0] = -S_3
    concat[1, 1] = 1 + S_3
    concat[1, 2] = S_4
    concat[2, 0] = epsilon * S_4
    concat[2, 1] = -epsilon * S_4
    concat[2, 2] = 1 + S_3 - rho * S_4

    return np.kron(concat, np.eye(3))
    