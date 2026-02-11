import numpy as np
from dh import dh_transform

def forward_kinematics(q):
    """
    Compute forward kinematics of UR5 using Standard DH parameters.

    Parameters:
    q : list of 6 joint angles [q1, q2, q3, q4, q5, q6] (in radians)

    Returns:
    List of transformation matrices:
    T_0^0, T_0^1, T_0^2, ..., T_0^6
    """

    # UR5 Standard DH parameters (in meters and radians)
    a = [0, -0.425, -0.392, 0, 0, 0]
    d = [0.0892, 0, 0, 0.1093, 0.0947, 0.0823]
    alpha = [np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]

    # Start with identity matrix (base frame)
    T = np.eye(4)

    # Store all intermediate transformations
    T_matrices = [T]

    # Sequentially multiply each joint transformation
    for i in range(6):
        T = T @ dh_transform(q[i], d[i], a[i], alpha[i])
        T_matrices.append(T)

    return T_matrices
