"""PID controller for ballbot balance testing."""
import numpy as np
import torch


class PID:
    """
    PID controller for ballbot balance testing.
    
    Only used for sanity checks after install. This is a classical control
    method, not part of the RL system.
    
    Attributes:
        k_p: Proportional gain
        k_i: Integral gain
        k_d: Derivative gain
        dt: Time step
        integral: Accumulated integral error
        prev_err: Previous error for derivative calculation
    """

    def __init__(self, dt, k_p, k_i, k_d):
        """
        Initialize PID controller.
        
        Args:
            dt: Time step
            k_p: Proportional gain
            k_i: Integral gain
            k_d: Derivative gain
        """
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.dt = dt

        self.integral = torch.zeros(2)
        self.prev_err = torch.zeros(2)

        self.err_hist = []
        self.integral_hist = []
        self.derivative_hist = []

        self.return_in_pitch_roll_space = False

    def act(self, R_mat: torch.tensor, setpoint_r=0, setpoint_p=0):
        """
        Compute PID control action.
        
        Args:
            R_mat: Rotation matrix tensor
            setpoint_p: Pitch target
            setpoint_r: Roll target
            
        Returns:
            Tuple of (control_action, angle_in_degrees)
            - control_action: 3D motor commands or 2D pitch/roll commands
            - angle_in_degrees: Current tilt angle
        """

        with torch.no_grad():

            gravity = torch.tensor([0, 0, -1.0]).float().reshape(3, 1)
            gravity_local = R_mat.T.mm(gravity).reshape(3)
            up_axis_local = torch.tensor([0, 0, 1]).float()

            #all in local coordinates
            error_vec_2d = torch.zeros(2)

            roll = torch.atan2(R_mat[2, 1], R_mat[2, 2])
            pitch = torch.atan2(-R_mat[2, 0],
                                torch.sqrt(R_mat[2, 1]**2 + R_mat[2, 2]**2))

            error_vec_2d[0] = setpoint_p - pitch
            error_vec_2d[1] = setpoint_r - roll

            self.integral += error_vec_2d * self.dt
            derivative = (error_vec_2d - self.prev_err) / self.dt

            u = self.k_p * error_vec_2d + self.k_i * self.integral + self.k_d * derivative

            self.prev_err = error_vec_2d

            angle_in_degrees = torch.acos(
                up_axis_local.dot(-gravity_local)).item() * 180 / np.pi

            self.err_hist.append(error_vec_2d.reshape(1, 2).numpy())

            up_axis_global = R_mat.mm(up_axis_local.reshape(3, 1))

            if self.return_in_pitch_roll_space:
                return u, angle_in_degrees
            else:  #return in motor space
                ctrl_c = torch.zeros(3)
                # Convert to motor space (120 degree spacing)
                ctrl_c[0] = u[1] * np.cos(np.deg2rad(0)) + u[0] * np.sin(np.deg2rad(0))
                ctrl_c[1] = u[1] * np.cos(np.deg2rad(120)) + u[0] * np.sin(np.deg2rad(120))
                ctrl_c[2] = u[1] * np.cos(np.deg2rad(240)) + u[0] * np.sin(np.deg2rad(240))
                ctrl_c = torch.clamp(ctrl_c, min=-10, max=10)

                return ctrl_c, angle_in_degrees

