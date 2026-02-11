import numpy as np
from fk import forward_kinematics
from jacobian import compute_jacobian

def inverse_kinematics(q_init, target_T, max_iters=1000, tol=1e-4):
    """
    Numerical IK for full pose (position + orientation)
    using Jacobian pseudo-inverse.
    """

    q = np.array(q_init, dtype=float)

    for _ in range(max_iters):

        T_matrices = forward_kinematics(q)
        T = T_matrices[-1]

        # Current position and rotation
        p = T[0:3, 3]
        R = T[0:3, 0:3]

        # Desired position and rotation
        p_d = target_T[0:3, 3]
        R_d = target_T[0:3, 0:3]

        # Position error
        e_p = p_d - p

        # Orientation error
        R_err = R_d @ R.T
        e_o = 0.5 * np.array([
            R_err[2,1] - R_err[1,2],
            R_err[0,2] - R_err[2,0],
            R_err[1,0] - R_err[0,1]
        ])

        # Combine position and orientation error
        error = np.concatenate((e_p, e_o))

        # Check convergence
        if np.linalg.norm(error) < tol:
            return q, True

        # Compute full Jacobian
        J = compute_jacobian(q)

        # Pseudo-inverse
        J_pinv = np.linalg.pinv(J)

        # Joint update
        dq = J_pinv @ error

        q = q + dq
        # Wrap angles to [-pi, pi]
        q = (q + np.pi) % (2 * np.pi) - np.pi


    return q, False
