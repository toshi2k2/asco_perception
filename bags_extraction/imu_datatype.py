# from typing import Tuple

# Orientation = Tuple[float, float, float, float]
# OrientationCov = Tuple[float, float, float, float, float, float, float, float, float]
# AngularVelocity = Tuple[float, float, float]
# AngularVelocityCov = Tuple[float, float, float, float, float, float, float, float, float]
# LinearAcceleration = Tuple[float, float, float]
# LinearAccelerationCov = Tuple[float, float, float, float, float, float, float, float, float]
# imu = Tuple[Orientation, OrientationCov, AngularVelocity, AngularVelocityCov, LinearAcceleration, LinearAccelerationCov, int]

class Imu:
    def __init__(self, ow, ox, oy, oz, oc, ax, ay, az, ac, lx, ly, lz, lc, t):
        super().__init__()
        self.orientation = (ow, ox, oy, oz)
        self.orientation_cov = oc
        self.angular_velocity = (ax, ay, az)
        self.angular_velocity_cov = ac
        self.linear_acceleration = (lx, ly, lz)
        self.linear_acceleration_cov = lc
        self.timestamp = t  # nanoseconds
