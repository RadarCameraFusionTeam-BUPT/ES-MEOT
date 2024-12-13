import numpy as np
from math import *
from .parameters import *


def lineIntersection(a, b, c, d):
    a, b, c, d = np.array(a), np.array(b), np.array(c), np.array(d)
    denominator = np.cross(b-a, d-c)
    if abs(denominator) < 1e-6:
        return np.mean(np.stack((a, b, c, d)), axis=0)
    x = a+(b-a)*(np.cross(c-a, d-c)/denominator)
    return x


def ccw(p, a, b):
    p, a, b = np.array(p), np.array(a), np.array(b)
    return np.cross(a-p, b-p)


def CalMiddle(a, b, c, d):
    if ccw(a, b, c)*ccw(a, b, d) < 0:
        return lineIntersection(a, b, c, d)
    elif ccw(a, c, b)*ccw(a, c, d) < 0:
        return lineIntersection(a, c, b, d)
    else:
        return lineIntersection(a, d, b, c)


def CalBackMiddle(kps):
    kps = np.array(kps)
    if len(kps) == 0:
        return None
    valid_idx = kps[:, 0]

    left_kp_id = np.array([5, 14, 7, 8])
    right_kp_id = np.array([4, 15, 19, 18])

    left_in_valid = np.isin(left_kp_id, valid_idx)
    right_in_valid = np.isin(right_kp_id, valid_idx)

    in_valid_id = np.where(np.logical_and(
        left_in_valid, right_in_valid) == True)[0]
    if len(in_valid_id) == 0:
        return None
    return (kps[np.where(valid_idx == left_kp_id[in_valid_id[0]])][0, 1:] + kps[np.where(valid_idx == right_kp_id[in_valid_id[0]])][0, 1:]) / 2

def CalBottomMiddle(kps):
    kps = np.array(kps)
    if len(kps) == 0:
        return None
    bottom_id = np.array([24, 25, 26, 27])
    kps_id_for_bottom = np.where(np.isin(kps[:, 0], bottom_id))[0]
    bottom = kps[kps_id_for_bottom]
    if len(bottom) < 4:
        return None
    return CalMiddle(bottom[0, 1:], bottom[1, 1:], bottom[2, 1:], bottom[3, 1:])


def dis(p, q):
    return np.sqrt(np.sum((p-q)**2))


def fpq(p, q, eps):
    tmp = dis(p, q)
    if tmp > eps:
        return 0
    else:
        return 1-tmp/eps


def CATS(t1, t2, tao, eps):
    totalScore = 0
    from collections import deque
    q = deque()
    idx = 0
    for p in t1:
        while idx < len(t2) and t2[idx]['frame'] <= p['frame']+tao:
            q.append(t2[idx])
            idx += 1
        while len(q) > 0 and q[0]['frame'] < p['frame']-tao:
            q.popleft()

        clueScore = 0
        for qq in q:
            clueScore = max(clueScore, fpq(p['pos'], qq['pos'], eps))
        totalScore += clueScore
    return totalScore/len(t1)

def rotation_vectors_to_matrices(rot_vecs):
    rot_vecs = np.atleast_2d(rot_vecs).astype(float)
    norms = np.linalg.norm(rot_vecs, axis=1)
    mask = norms > eps
    n = len(rot_vecs)
    n_mask = np.count_nonzero(mask)

    k = np.zeros_like(rot_vecs)
    k[mask] = rot_vecs[mask] / norms[mask, np.newaxis]

    K = np.zeros((len(rot_vecs), 3, 3))
    zeros_n_mask_1 = np.zeros(n_mask)
    K[mask, 0, 1] = -k[mask, 2]
    K[mask, 0, 2] = k[mask, 1]
    K[mask, 1, 0] = k[mask, 2]
    K[mask, 1, 2] = -k[mask, 0]
    K[mask, 2, 0] = -k[mask, 1]
    K[mask, 2, 1] = k[mask, 0]

    theta = norms
    R = np.eye(3) + np.sin(theta)[:, np.newaxis, np.newaxis] * K + \
        (1 - np.cos(theta))[:, np.newaxis, np.newaxis] * K @ K
    return R

def rotation_matrics_to_vectors(rot_matrics):
    rot_matrics = np.expand_dims(rot_matrics, axis=0) if rot_matrics.ndim == 2 else rot_matrics

    # Compute the trace of each rotation matrix
    trace = np.trace(rot_matrics, axis1=-2, axis2=-1)
    
    # Compute the angle of rotation
    theta = np.arccos((trace - 1) / 2)
    
    # Compute the normalization factor, handling division by zero
    mask_zero = np.isclose(theta, 0)
    norm_factor = np.zeros_like(theta)
    norm_factor[~mask_zero] = 0.5 / np.sin(theta[~mask_zero])
    
    # Initialize the rotation vectors array
    rot_vectors = np.zeros((rot_matrics.shape[0], 3))
    
    # Compute rotation vectors
    rot_vectors[:, 0] = (rot_matrics[:, 2, 1] - rot_matrics[:, 1, 2]) * norm_factor * theta
    rot_vectors[:, 1] = (rot_matrics[:, 0, 2] - rot_matrics[:, 2, 0]) * norm_factor * theta
    rot_vectors[:, 2] = (rot_matrics[:, 1, 0] - rot_matrics[:, 0, 1]) * norm_factor * theta
    
    return rot_vectors


def normalize_weight(log_weights):
    sum_bern_log_weights = np.sum(log_weights, axis=0)
    C = np.max(sum_bern_log_weights)
    normalize_log_weight = sum_bern_log_weights - np.log(np.sum(np.exp(sum_bern_log_weights - C))) - C
    return np.exp(normalize_log_weight)

def get_J(v):
    v = np.atleast_2d(v)
    norm_v = np.linalg.norm(v, axis=1)
    skew = skew_symmetry(v)
    # mark norm_v not zero
    mask = norm_v > eps

    J = np.zeros((v.shape[0], 3, 3))
    J[mask] = np.eye(3) + ((1 - np.cos(norm_v[mask])) / norm_v[mask]**2)[:, np.newaxis, np.newaxis] * skew[mask] + ((norm_v[mask] - np.sin(norm_v[mask])) / norm_v[mask]**3)[:, np.newaxis, np.newaxis] * skew[mask] @ skew[mask]
    J[~mask] = np.eye(3) + 1 / 2 * skew[~mask] + 1 / 6 * skew[~mask] @ skew[~mask]

    return J

def skew_symmetry(v):
    if v.ndim == 1:
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
    else:
        result = np.zeros((*v.shape[:-1], 3, 3))
        result[..., 0, 1] = -v[..., 2]
        result[..., 0, 2] = v[..., 1]
        result[..., 1, 0] = v[..., 2]
        result[..., 1, 2] = -v[..., 0]
        result[..., 2, 0] = -v[..., 1]
        result[..., 2, 1] = v[..., 0]
        return result


def exp_skew_v(v):
    v = np.atleast_2d(v)
    norm_v = np.linalg.norm(v, axis=1)
    n = len(v)

    mask = norm_v > eps
    ret = np.tile(np.eye(3), (n, 1, 1))
    n_mask = np.count_nonzero(mask)

    if n_mask > 0:
        skew = skew_symmetry(v[mask] / norm_v[mask, np.newaxis])
        ret[mask] += np.sin(norm_v[mask])[:, np.newaxis, np.newaxis] * skew + \
            (1 - np.cos(norm_v[mask])[:, np.newaxis, np.newaxis]) * skew @ skew

    return ret


def Star(M):
    assert M.ndim == 2 or M.ndim == 3 or M.ndim == 4
    if M.ndim == 2:
        assert M.shape[0] == 3
        return np.concatenate(skew_symmetry(M.T), axis=1)
    elif M.ndim == 3:
        ret = skew_symmetry(M.transpose(0, 2, 1).reshape([-1, 3]))
        ret = ret.reshape([-1, 3, 3, 3]).transpose(0, 2, 1, 3).reshape([-1, 3, 9])
        return ret
    else:
        n, m = M.shape[0], M.shape[1]
        ret = skew_symmetry(M.transpose(0, 1, 3, 2).reshape([-1, 3]))
        ret = ret.reshape([n, m, 3, 3, 3]).transpose(0, 1, 3, 2, 4).reshape([n, m, 3, 9])
        return ret

def Sqrt(M):
    assert M.ndim >= 2
    if M.ndim == 2:
        try:
            return np.linalg.cholesky(M)
        except:
            return np.zeros_like(M)
    else:
        ### M = d @ d.T ###
        d = np.zeros_like(M)
        d[..., 0, 0] = np.sqrt(M[..., 0, 0])
        d[..., 1, 0] = M[..., 1, 0] / d[..., 0, 0]
        d[..., 1, 1] = np.sqrt(M[..., 1, 1] - d[..., 1, 0]**2)
        d[..., 2, 0] = M[..., 2, 0] / d[..., 0, 0]
        d[..., 2, 1] = (M[..., 2, 1] - d[..., 2, 0] * d[..., 1, 0]) / d[..., 1, 1]
        d[..., 2, 2] = np.sqrt(M[..., 2, 2] - d[..., 2, 0]**2 - d[..., 2, 1]**2)
        d = np.transpose(d, (*range(d.ndim - 2), d.ndim - 1, d.ndim - 2))
        return d

def diag_n(M, n):
    assert M.ndim >= 2
    height, width = M.shape[-2:]
    ret = np.zeros((*M.shape[:-2], height * n, width * n))
    for i in range(n):
        ret[..., i * height:(i + 1) * height, i * width:(i + 1) * width] = M
    return ret

def Pentacle(M):
    assert M.ndim >= 2
    if M.ndim == 2:
        return M.flatten('F')
    else:
        permute_index = list(range(M.ndim))
        permute_index[-1], permute_index[-2] = permute_index[-2], permute_index[-1]
        M_trans = M.transpose(permute_index)
        
        return M_trans.reshape([*M_trans.shape[:-2], -1])