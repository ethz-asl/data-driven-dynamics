__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

from ...tools import symmetric_logistic_sigmoid


class LinearPlateAeroModel():
     def __init__(self, stall_angle):
        self.stall_angle = stall_angle

    def compute_aero_features(self, v_airspeed, angle_of_attack):
        v_xz = math.sqrt(v_airspeed[0]**2 + v_airspeed[2]**2)
        F_xz_airspeed_frame = np.zeros((3, 4))
        F_xz_airspeed_frame[0, 3] = 1
        F_xz_airspeed_frame[2, 0] = (
            1 - symmetric_logistic_sigmoid(angle_of_attack, self.stall_angle))*angle_of_attack
        F_xz_airspeed_frame[2, 1] = (1 - symmetric_logistic_sigmoid(angle_of_attack, self.stall_angle))
        F_xz_airspeed_frame[2, 2] = 2*symmetric_logistic_sigmoid(angle_of_attack, self.stall_angle)
             *math.sin(angle_of_attack)*math.cos(angle_of_attack)
        F_xz_airspeed_frame = F_xz_airspeed_frame*v_xz**2
