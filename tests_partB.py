"""
Autonomous Robots
"""
import numpy as np
from math import radians
from matplotlib import pyplot as plt
from Transforms import rotation_matrix, get_transform
from Transforms import quaternion_from_R, R_from_quaternion
from Transforms import quaternion_slerp
from utils import DisplayFrame

######## Part B #######################
t0 = np.array([0., 0., 0.]) 
R0 = np.identity(3)
T0 = np.identity(4) # origin

theta_init = radians(60)
t1 = np.array([-5., -5., -5.]) 
K1 = np.array([0., 0., 1.]) # axis of rotation
R1 = rotation_matrix(theta_init, K1)
T1 = get_transform(R1, t1) # frame 1
"""
print(T1)=
[[ 0.5       -0.8660254  0.        -5.       ]
 [ 0.8660254  0.5        0.        -5.       ]
 [ 0.         0.         1.        -5.       ]
 [ 0.         0.         0.         1.       ]]
"""


theta_final = radians(45)
t2 = np.array([-15., -15., -15.])
K2 = np.array([0., 1., 1.]) # axis of rotation
R2 = rotation_matrix(theta_init, K2)
T2 = get_transform(R2, t2) # frame 2
"""
print(T2)=
[[  0.5         -0.61237244   0.61237244 -15.        ]
 [  0.61237244   0.75         0.25       -15.        ]
 [ -0.61237244   0.25         0.75       -15.        ]
 [  0.           0.           0.           1.        ]]
"""


## draw the frames
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax = DisplayFrame(R0, t0, ax, came_scale=1) # origin
ax = DisplayFrame(R1, t1, ax, came_scale=2) # frame 1
ax = DisplayFrame(R2, t2, ax, came_scale=2) # frame 2

## draw intermediate frames by interpolation
num_levels = 50

# rotation interpolation in quaternion space
q1 = quaternion_from_R(R1)
q2 = quaternion_from_R(R2)
q_interp = quaternion_slerp(q1, q2, levels=num_levels)

# translation is easy to interpolate
x_interp = np.linspace(t1[0], t2[0], num_levels)
y_interp = np.linspace(t1[1], t2[1], num_levels)
z_interp = np.linspace(t1[2], t2[2], num_levels)

for i in range(num_levels):
    q = q_interp[i]
    R = R_from_quaternion(q)
    t = np.array([x_interp[i], y_interp[i], z_interp[i]])
    ax = DisplayFrame(R, t, ax, came_scale=1) # frame i
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
"""
num_levels = 10 # change to 50/100 to see the dense pattern
"""

