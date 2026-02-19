import rclpy
import numpy as np
import pickle


class GeneratePwm():
    '''
    This class apply an open-loop model to convert 
    acceleration and steering angle to PWM that can
    be read by the ESC and servo
    '''
    def __init__(self, min_throttle, max_throttle, model_path):
        '''
        Constructor for the GeneratePwm class
        '''        
        # Read the parameters from the parameter server
        self.min_throttle = min_throttle
        self.max_throttle = max_throttle
        self.model_path = model_path

        # Define the open-loop model
        self.mlp_model = pickle.load(open(self.model_path, 'rb'))
        
    def convert(self, accel: float, steer: float, v: float):
        '''parameters
        convert the acceleration and steering angle to PWM given the current state
        Parameters:
            accel: float, linear acceleration of the robot [m/s^2]
            steer: float, steering angle of the robot [rad]
            state: State2D, current state of the robot
        '''

        # Do not allow the car to go over 3m/s
        if v > 3:
            accel = min(accel, 0)
            v_bounded = 3
        else: 
            v_bounded = v
        
        # negative pwm means turn left (positive steering angle)
        steer_pwm = -np.clip(steer/0.37, -1, 1)
        accel_bounded = np.sign(accel)*min(abs(accel), 2+v)

        # Generate Input vector
        input = np.array([[accel_bounded, v_bounded, np.abs(steer_pwm)]])
        
        # check nan
        if np.any(np.isnan(input)):
            self.get_logger().warn("Contain NAN in control!")
            return self.min_throttle, steer_pwm

        # convert the acceleration and steering angle to PWM
        d = self.mlp_model.predict(input)[0]
        
        # clip the throttle to the maximum and minimum throttle
        throttle_pwm = np.clip(d, self.min_throttle, self.max_throttle)
        
        # Composite the throttle for low speed
        if v<0.2:
            throttle_pwm += np.abs(steer_pwm)*0.04
            
        return throttle_pwm, steer_pwm
        
        
        