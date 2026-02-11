from ik import inverse_kinematics
from fk import forward_kinematics
import numpy as np
from verify import verify_solution


# Desired full pose (identity orientation + position)
target_T = np.eye(4)
target_T[0:3, 3] = [-0.6, -0.2, 0.2]

q_init = [0, 0, 0, 0, 0, 0]

# Solve IK
q_sol, success = inverse_kinematics(q_init, target_T)

print("\nIK Success:", success)
print("Joint Angles:")
print(q_sol)

# Verify solution
T_verify = forward_kinematics(q_sol)[-1]
final_position = T_verify[0:3, 3]

print("\nFinal Position after IK:")
print(final_position)

print("\nFinal Position Error:")
print(np.linalg.norm(target_T[0:3, 3] - final_position))

verify_solution(q_sol, target_T)
