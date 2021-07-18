#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


fig = plt.figure("Rotor Feature Computation")

ax1 = fig.add_subplot(1, 2, 1, projection='3d')

X = np.arange(0, 1, 0.05) # Actuator Inputs
Y = np.arange(0, 10, 0.05) # Input Airspeed
X, Y = np.meshgrid(X, Y)

### Compute Rotor Thrust From RotorFeatures
# Compute rotor thrust
thrust_lin = -0.1425458591033593
thrust_quad = 4.003013943813853
airdensity = 1.18
prop_diameter = 1.0

Z = (thrust_lin * np.multiply(Y, X) + thrust_quad *np.power(X, 2) * prop_diameter) * airdensity * np.power(prop_diameter, 3)

surf = ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax1.title.set_text("Estimated Thrust")
ax1.set_zlim([0, 6])
ax1.set_xlabel('actuator inputs')
ax1.set_ylabel('parallel airspeed')
ax1.view_init(elev=45., azim=225.)

ax = fig.add_subplot(1, 2, 2, projection='3d', sharex=ax1, sharey=ax1)

### Compute Rotor Thrust From GazeboMotorModelPlugin
# Parameters from default iris model: https://github.com/PX4/PX4-SITL_gazebo/blob/55e479aa80b57da850e307b0105baf0f262ec284/models/iris/iris.sdf.jinja#L353
# Rotor thrust calculations from GazeboMotorModelPlugin: https://github.com/PX4/PX4-SITL_gazebo/blob/55e479aa80b57da850e307b0105baf0f262ec284/src/gazebo_motor_model.cpp#L224
motor_constant = 5.84e-06
moment_constant = 0.06
zeroposition_armed = 100
input_scaling = 1000
rotordrag_coefficient = 0.000175
rollingmoment_coefficient = 1e-06
airdensity = 1.18
prop_diameter = 1.0

# Compute Rotor Thrust From RotorFeatures
Z = (1.0 - np.multiply(Y, 1/25.0)) * np.power((X * input_scaling + zeroposition_armed), 2) * motor_constant

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.title.set_text("Simulated Thrust")
ax.set_zlim([0, 6])
ax.set_xlabel('actuator inputs')
ax.set_ylabel('parallel airspeed')
ax.view_init(elev=45., azim=225.)



fig2, ax3 = plt.subplots()

actuator_input = np.arange(0, 1, 0.05) # Actuator Inputs
simulated_thrust = np.power((actuator_input * input_scaling + zeroposition_armed), 2) * motor_constant
ax3.plot(actuator_input, simulated_thrust, label='Simulated Thrust')

# Z = (thrust_lin * np.multiply(Y, X) + thrust_quad *np.power(X, 2) * prop_diameter) * airdensity * np.power(prop_diameter, 3)
estimated_thrust = (thrust_quad *np.power(actuator_input, 2) * prop_diameter) * airdensity * np.power(prop_diameter, 3)
ax3.plot(actuator_input, estimated_thrust, label='Estimated Thrust')
ax3.set_title("Rotor Thrust")
ax3.legend()
plt.grid()
plt.show()
