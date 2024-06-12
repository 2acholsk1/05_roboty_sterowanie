import numpy as np
from .controller import Controller
from models.manipulator_model import ManipulatorModel as MM


class MMAController(Controller):
    def __init__(self, Tp, kp, kd):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        self.kp = kp
        self.kd = kd
        self.models = [MM(Tp, 0.1, 0.05), MM(Tp, 0.01, 0.01), MM(Tp, 1.0, 0.3)]
        self.i = 0
        self.first_step = True

    def choose_model(self, x, u=np.array([0.0, 0.0])):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        q = x[:2]
        q_dot = x[2:]
        
        errors = np.array([])
        for model in self.models:
            model_prediction = model.M(x) @ u + model.C(x) @ q_dot[:, np.newaxis]
            errors = np.append(errors, np.sum(np.abs(q_dot - model_prediction)))
        self.i = np.argmin(errors)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        if self.first_step:
            self.choose_model(x)
            self.first_step = False
        else:
            self.choose_model(x, self.v)
        q = x[:2]
        q_dot = x[2:]
        # v = q_r_ddot # TODO: add feedback
        v = q_r_ddot + self.kd*(q_dot- q_r_dot) + self.kp*(q - q_r)
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        self.v = v[:, np.newaxis]
        return u
