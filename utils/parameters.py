import numpy as np
import os
import sys
from scipy.linalg import block_diag
from typing import List
from scipy.stats.distributions import chi2

############## parameters for TPMB-RCfusion #################
Ps = 0.99

# detection rate
Pd_r = 0.5
Pd_c = 0.8

# mean number of birth
lambda_b = 4

# mean number of radar detections
alpha = 15

# not detection rate
q_d_r = 1 - Pd_r + Pd_r * np.exp(-alpha)
q_d_c = 1 - Pd_c

# prune threshold of probability of Bernoulli component
threshold_r = 1e-4
# prune weight of mixture component in PPP
threshold_u = 1e-5
# recycle threshold of Bernoulli component
threshold_cycle = 1e-3

# number of associations for partitions and mb components
M = 5

# gating parameters
gate_3d = chi2.ppf(0.999, df=3)
gate_2d = chi2.ppf(0.999, df=2)
gate_1d = chi2.ppf(0.999, df=1)

# dbscan parameters
# minimize the generation of clutters
dbscan_eps, dbscan_min_samples = 2, 10

############## parameters for sigle object tracking #################
# If the angular velocity is less than eps
# it is considered 0
eps = 1e-9

# elastic coefficient
epsilon = 100

# damping factor
rho = 20

# time interval
dt = 1 / 15

# Process noise
W = block_diag(np.eye(3) * 0.1, np.eye(1) * 0.01,
               np.eye(3) * 0.1, np.eye(3) * 0.1, np.eye(3) * 0.1)
# W_vartheta = block_diag(np.eye(3) * 0.0, np.zeros((3, 3)), np.eye(3) * 0.2)
W_vartheta = np.eye(9) * 0.2

# Number of reflector points
N_T = 24

# Corner point idx
corner_id = np.array([24, 25, 26, 27, 28, 29, 30, 31])

# relationship of corners and extend
# corners_id 24, 25, 26, 27, 28, 29, 30, 31
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

# Number of corner points
N_c = len(corner_id)

# Measurement noise of radar
Q = np.diag([0.1, 0.3, 0.3])
Q_inv = np.linalg.inv(Q)
det_Q = np.linalg.det(Q)

# Measurement noise of pixel keypoints
V_c = np.eye(2) * 15
V_c_inv = np.linalg.inv(V_c)
det_V_c = np.linalg.det(V_c)

# Measurement noise of pseudo measurements
Q_rot = np.array([[0.1]])
Q_rot_inv = np.linalg.inv(Q_rot)

Q_model = np.eye(3) * 1.0
Q_model_inv = np.linalg.inv(Q_model)

Q_ground = np.array([[1e-2]])
Q_ground_inv = np.linalg.inv(Q_ground)

Q_pose = np.eye(3) * 0.1
Q_pose_inv = np.linalg.inv(Q_pose)

# Flip
D = np.diag([1.0, -1.0, 1.0])
flip_id = np.array([1, 0, 3, 2, 5, 4, 16, 19, 18, 17, 11, 10, 13, 12,
                   15, 14, 6, 9, 8, 7, 21, 20, 23, 22])

# Car heading direction
u_d = np.array([1.0, 0.0, 0.0])

# road vector (up)
# road_vector = np.array([0.002671761716973599, -0.9396342322901945, -0.3421700910333109])
# d_ground = 6.73240454224306

road_vector = np.array([-0.12566874626130037, -5.814241488307704, -1])
d_ground = 39.855495690989194

road_vector_norm = np.linalg.norm(road_vector)
road_vector /= road_vector_norm
d_ground /= road_vector_norm

# Number of iterations of VB
N_iter = 3

# Constant parameters
H_r = np.zeros((3, 13))
H_r[:, :3] = np.eye(3)
H_theta = np.zeros((3, 13))
H_theta[:, 4:7] = np.eye(3)
H_omega = np.zeros((3, 13))
H_omega[:, 7:10] = np.eye(3)
H_ext = np.zeros((3, 13))
H_ext[:, 10:13] = np.eye(3)

H_u = np.zeros((3, 9))
H_u[:, :3] = np.eye(3)
H_varpi = np.zeros((3, 9))
H_varpi[:, 3:6] = np.eye(3)