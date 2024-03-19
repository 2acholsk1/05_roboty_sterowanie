import numpy as np
from .controller import Controller


class PDDecentralizedController(Controller):
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO: Please implement me
        q_12 = x[:2]
        q_dot_12 = x[2:]
        u = q_d_ddot + self.kd*(q_dot_12 - q_d_dot) + self.kp*(q_12 - q_d)
        return u
 