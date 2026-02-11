import numpy as np
from fk import forward_kinematics
from jacobian import compute_jacobian

def verify_solution(q_sol, target_T):
    """
    Verify IK solution by checking:
    - Position error
    - Orientation error
    - Jacobian condition number
    """

    # Compute FK of solution
    T_final = forward_kinematics(q_sol)[-1]

    # Extract position and rotation
    p_final = T_final[0:3, 3]
    R_final = T_final[0:3, 0:3]

    p_target = target_T[0:3, 3]
    R_target = target_T[0:3, 0:3]

    # Position error
    pos_error = np.linalg.norm(p_target - p_final)

    # Orientation error
    R_err = R_target @ R_final.T
    ori_error = 0.5 * np.linalg.norm([
        R_err[2,1] - R_err[1,2],
        R_err[0,2] - R_err[2,0],
        R_err[1,0] - R_err[0,1]
    ])

    # Jacobian condition number
    J = compute_jacobian(q_sol)
    cond_number = np.linalg.cond(J)

    print("\n===== VERIFICATION RESULTS =====")
    print("Final Position:", p_final)
    print("Position Error:", pos_error)
    print("Orientation Error:", ori_error)
    print("Jacobian Condition Number:", cond_number)

    if pos_error < 1e-4 and ori_error < 1e-4:
        print("Verification: SUCCESS")
    else:
        print("Verification: FAILED")

    return pos_error, ori_error, cond_number
