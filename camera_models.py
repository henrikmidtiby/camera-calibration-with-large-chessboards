from math import cos, sin

import cv2
import numpy as np
from icecream import ic


class PinholeCameraModel:
    def __init__(self):
        self.K = None

    def rotationMatrix(self, ax, ay, az):
        Rx = np.array([[1, 0, 0], [0, cos(ax), -sin(ax)], [0, sin(ax), cos(ax)]])
        Ry = np.array([[cos(ay), 0, sin(ay)], [0, 1, 0], [-sin(ay), 0, cos(ay)]])
        Rz = np.array([[cos(az), -sin(az), 0], [sin(az), cos(az), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        return R

    def externalMatrix(self, x):
        ax = x[0]
        ay = x[1]
        az = x[2]
        tx = x[3]
        ty = x[4]
        tz = x[5]

        R = self.rotationMatrix(ax, ay, az)
        Mext = np.concatenate((R, np.array([[tx], [ty], [tz]])), 1)
        return Mext

    def fProject(self, camera_pose, points):
        # Function structure from https://www.youtube.com/watch?v=kq3c6QpcAGc
        Mext = self.externalMatrix(camera_pose)

        ph = self.K @ Mext @ points
        ph[0, :] = np.true_divide(ph[0, :], ph[2, :])
        ph[1, :] = np.true_divide(ph[1, :], ph[2, :])
        ph = ph[0:2, :]

        ph = ph.reshape((1, -1))
        return ph


class CameraModel:
    def __init__(self):
        self.K = None
        self.dist = None

    def fProject(self, camera_pose, points):
        ph, _ = cv2.projectPoints(
            points,
            rvec=camera_pose[:3],
            tvec=camera_pose[3:],
            cameraMatrix=self.K,
            distCoeffs=self.dist,
        )
        ph = ph.reshape((1, -1))
        return ph


class FishEyeCameraModel:
    def __init__(self):
        self.K = None
        self.dist = None

    def fProject(self, camera_pose, points: np.ndarray):
        points = np.expand_dims(points, 0)
        ph, _ = cv2.fisheye.projectPoints(
            points,
            rvec=camera_pose[:3],
            tvec=camera_pose[3:],
            K=self.K,
            D=self.dist,
        )
        ph = ph.reshape((1, -1))
        return ph
