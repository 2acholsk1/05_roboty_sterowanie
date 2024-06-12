import numpy as np
from models.manipulator_model import ManipulatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp, kp, kd):
        self.model = ManipulatorModel(Tp)
        self.kp = kp
        self.kd = kd

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        q = x[:2]
        q_dot = x[2:]
        u = q_r_ddot + self.kd*(q_dot- q_r_dot) + self.kp*(q - q_r)
        v = self.model.M(x) @ u + self.model.C(x) @ q_dot 
        return v