import numpy as np
from fk import forward_kinematics

def compute_jacobian(q):
    """
    Compute the 6x6 Geometric Jacobian matrix for UR5.

    For revolute joints:
    Linear velocity part:  Jv_i = z_i x (p_e - p_i)
    Angular velocity part: Jw_i = z_i

    Parameters:
    q : list of 6 joint angles (in radians)

    Returns:
    6x6 Jacobian matrix
    """

    # Get all transformation matrices
    T_matrices = forward_kinematics(q)

    # End-effector position
    p_e = T_matrices[-1][0:3, 3]

    # Initialize Jacobian matrix
    J = np.zeros((6, 6))

    for i in range(6):
        T_i = T_matrices[i]

        # Position of joint i
        p_i = T_i[0:3, 3]

        # Z-axis of joint i (axis of rotation)
        z_i = T_i[0:3, 2]

        # Linear velocity contribution
        J_v = np.cross(z_i, (p_e - p_i))

        # Angular velocity contribution
        J_w = z_i

        # Fill Jacobian matrix
        J[0:3, i] = J_v
        J[3:6, i] = J_w

    return J
