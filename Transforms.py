import math
import numpy as np
from libs_hh2.utils import unit_vector

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0


def t_from_T(matrix):
    """Return translation vector from transformation matrix.
    """
    return np.array(matrix, copy=False)[:3, 3].copy()


def rotation_matrix(angle, direction):
    """Return matrix to rotate about axis defined by point and direction.
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array([[ 0.0,  -direction[2],  direction[1]],
                      [ direction[2],  0.0,  -direction[0]],
                      [-direction[1],  direction[0],  0.0]])
    return R


def get_transform(R, t):
    """Return matrix to rotate about axis defined by point and direction.
    """
    T = np.identity(4)
    T[:3, :3] = R
    t = np.array(t[:3], dtype=np.float64, copy=False)
    T[:3, 3] = t
    return T

def slerp(q0, q1, t):
    """Finding intermediate frames by interpolation"""
    d = np.dot(q0, q1)
    if d < 0.0:
        q1 = -q1
        d = -d

    if d > 0.9995:
        q_t = (1.0 - t) * q0 + t * q1
        return unit_vector(q_t)

    theta = math.acos(d)
    q_t = (math.sin((1 - t) * theta) / math.sin(theta)) * q0 + (math.sin(t * theta) / math.sin(theta)) * q1

    return unit_vector(q_t)


def quaternion_slerp(quat0, quat1, levels=5):
    """Return spherical linear interpolation between two quaternions.
    """
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    d = np.dot(q0, q1)
    angle = math.acos(d)
    if abs(angle) < _EPS:
        return q0

    all_q = [q0] # the first one

    # your code: find all intermediate quaternions
    for i in range(1, levels):
        t = i / levels
        q_t = slerp(q0, q1, t)
        all_q.append(q_t)

    all_q.append(q1) # the last one
    return all_q


def quaternion_about_axis(angle, axis):
    """Return quaternion for rotation about axis.
    """
    q = np.array([0.0, axis[0], axis[1], axis[2]])
    qlen = np.linalg.norm(q)
    if qlen > _EPS:
        q *= math.sin(angle/2.0) / qlen
    q[0] = math.cos(angle/2.0)
    return q


def R_from_quaternion(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    """
    q = np.array(quaternion, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        R = np.identity(3)
    else:
       # your code: perform the calculation for R
        qx, qy, qz, qw = quaternion
        R = np.array([[(1 - 2 * qy ** 2 - 2 * qz ** 2), 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                      [2 * qx * qy + 2 * qz * qw, (1 - 2 * qx ** 2 - 2 * qz ** 2), 2 * qy * qz - 2 * qx * qw],
                      [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, (1 - 2 * qx ** 2 - 2 * qy ** 2)]])
    return R


def quaternion_from_R(matrix):
    """Return quaternion from rotation matrix.
        Quaternion is in (x, y, z, w) format.
    """
    M = np.array(matrix, dtype=np.float64, copy=False)
    m00 = M[0, 0]; m01 = M[0, 1]; m02 = M[0, 2]
    m10 = M[1, 0]; m11 = M[1, 1]; m12 = M[1, 2]
    m20 = M[2, 0]; m21 = M[2, 1]; m22 = M[2, 2]
    
    # your code: perform the calculation for q
    qw, qx, qy, qz = 0, 0, 0, 0

    R_trace = m00 + m11 + m22
    if R_trace > 0:
        S = np.sqrt(R_trace + 1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    return np.array([qx, qy, qz, qw]) # in (x, y, z, SCALAR) format


def compund_transform(*matrices):
    """Return concatenation of series of transformation matrices.
    """
    T = np.identity(4)
    for M in matrices:
        T = np.dot(T, M)
    return T



    