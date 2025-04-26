from scipy.spatial.transform import Rotation as R
import math
import numpy as np

theta  = math.radians(90.2)
axis = np.array([0.016, -0.959, 0.282])

euler = R.from_rotvec(theta * axis, degrees=False).as_euler('XYZ', degrees=True)
matrix = R.from_rotvec(theta * axis, degrees=False).as_matrix()
print(euler)
a = 0

theta  = math.radians(90.2)
axis = np.array([0.016, -0.959, 0.282])
axis = axis / np.linalg.norm(axis)

# rotation matrix from axis-angle
K = np.array([[0,          -axis[2],  axis[1]],
              [axis[2],     0,        -axis[0]],
              [-axis[1],    axis[0],   0       ]])
Ro = np.eye(3) + math.sin(theta)*K + (1-math.cos(theta))*(K@K)

# extract intrinsic XYZ Euler
if abs(Ro[0,2]) != 1.0:              # |sβ| != 1
    Y =  math.asin( Ro[0,2])         # β
    X =  math.atan2(-Ro[1,2], Ro[2,2])   # α
    Z =  math.atan2(-Ro[0,1], Ro[0,0])   # γ
else:                                # gimbal lock
    Z = 0
    if Ro[0,2] < 0:                  # β = +90°
        Y =  math.pi/2
        X =  Z + math.atan2( Ro[1,0], Ro[1,1])
    else:                            # β = −90°
        Y = -math.pi/2
        X = -Z + math.atan2(-Ro[1,0],-Ro[1,1])

print(np.degrees([X, Y, Z]))

X_matrix = R.from_euler(seq="X", angles=X, degrees=False).as_matrix()
Y_matrix = R.from_euler(seq="Y", angles=Y, degrees=False).as_matrix()
Z_matrix = R.from_euler(seq="Z", angles=Z, degrees=False).as_matrix()
rot_matrix = np.matmul(X_matrix, np.matmul(Y_matrix, Z_matrix))
adw = -0

A = [ 75.05001107, -72.69325195,  90.54109638]   # deg
B = [-73.36316251, -74.53271874,  90.69473407]   # deg

R_A = R.from_euler('XYZ', A, degrees=True).as_matrix()
R_B = R.from_euler('XYZ', B, degrees=True).as_matrix()

print(np.allclose(R_A, R_B))
print(np.max(np.abs(rot_matrix-matrix)))
dwqa = 0


