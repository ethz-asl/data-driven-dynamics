from src.tools import quaternion_to_rotation_matrix


R0 = quaternion_to_rotation_matrix([0.705, 0, 0, 0.705])
print(R0)

R_forward = quaternion_to_rotation_matrix([0.745, 0.072, -0.081, 0.66])
print(R_forward)
