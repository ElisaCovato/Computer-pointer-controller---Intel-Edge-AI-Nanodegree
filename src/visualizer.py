import cv2
import numpy as np
from math import cos, sin, pi



class ShowPreview:
    '''
    This class is used to display the output of intermediate models
    '''
    def __init__(self, frame, flip):

        self.frame = frame
        self.flip = flip

        self.prev_w = 600
        self.prev_h = 600




    def draw_face_box(self, xmin, ymin, xmax, ymax):
        '''
        Draw a face box around the face detection
        '''
        cv2.rectangle(self.frame, (xmin, ymin), (xmax, ymax),
                      (145, 50, 255), 2)


    def draw_eye_box(self, eyes_coord, face_coords, square_size):
        '''
        Draw a box around the eyes
        '''

        x_min, y_min = eyes_coord[0] + face_coords[0] - square_size, \
                       eyes_coord[1] + face_coords[1] - square_size
        x_max, y_max = eyes_coord[0] + face_coords[0] + square_size, \
                       eyes_coord[1] + face_coords[1] + square_size

        cv2.rectangle(self.frame, (x_min, y_min), (x_max, y_max), (200, 10, 10), 2)


    def draw_head_pose(self, yaw, pitch, roll, anchor_point):
        '''
        Draw head pose directions axis:
            Green: Y-axis / Yaw
            Red: X-axis / Pitch
            Blue: Z-axis / Roll
        Angles are in Tait-Bryan format.
        '''
        # Transform Tait-Bryan angles to radians
        pitch = pitch * pi / 180
        yaw = yaw * pi / 180
        roll = roll * pi / 180

        # Set scale and focal length
        self.scale = 50
        self.focal_len = 950.0

        # Create projectiona axis and get projected coordinates for the angles
        axis_system = self.create_projection_axis(yaw, pitch, roll)
        projec_coords = self.project_angles(axis_system, anchor_point)
        xp, yp, zp, zp1 = projec_coords

        # Draw axis on frame
        cv2.line(self.frame, anchor_point, xp, (0, 0, 255), 2)  # x-axis
        cv2.line(self.frame, anchor_point, yp, (0, 255, 0), 2)  # y-axis
        cv2.line(self.frame, zp1, zp, (255, 0, 0), 2)  # z-axis
        cv2.circle(self.frame, zp, 3, (255, 0, 0), 2) # front of z axis

    def draw_eye_gaze(self, eyes_coord, face_coords, gaze, length):
        '''
        Draw arrows to indicate eye gaze
        '''
        eye_xcenter = int(eyes_coord[0] + face_coords[0])
        eye_ycenter = int(eyes_coord[1] + face_coords[1])

        gaze_x = (gaze[0] * length[0]) + eye_xcenter
        gaze_y = (-gaze[1] * length[1]) + eye_ycenter
        gp = (int(gaze_x), int(gaze_y))

        cv2.arrowedLine(self.frame, (eye_xcenter, eye_ycenter), gp, (230, 216, 173), 2)

    def create_projection_axis(self, yaw, pitch, roll):
        '''
        Use head pose angles to create the axis system for the frame.
        The axis will be used to project the angles and get their cartesian coords with respect to
        frame axis system.
        Input: angles
        Return: axis system
        '''
        # Create axis where to project angles
        # ref1: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
        # ref2: https://github.com/opencv/open_model_zoo/blob/master/demos/interactive_face_detection_demo/visualizer.cpp
        if self.flip: # if the image needs to be flipped, the x axis needs to be flipped as well
            x_flip = -1
        else:
            x_flip = 1
        xaxis = np.array(([x_flip * self.scale, 0, 0]), dtype='float32').reshape(3, 1)
        yaxis = np.array(([0, -1 * self.scale, 0]), dtype='float32').reshape(3, 1)
        zaxis = np.array(([0, 0, -1 * self.scale]), dtype='float32').reshape(3, 1)
        zaxis1 = np.array(([0, 0, 1 * self.scale]), dtype='float32').reshape(3, 1)

        # Translation matrix
        t = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
        t[2] = self.focal_len

        # Construct rotation matrices according to https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19770024290.pdf
        Rx = np.array([[1, 0, 0],
                       [0, cos(pitch), -sin(pitch)],
                       [0, sin(pitch), cos(pitch)]])
        Ry = np.array([[cos(yaw), 0, -sin(yaw)],
                       [0, 1, 0],
                       [sin(yaw), 0, cos(yaw)]])
        Rz = np.array([[cos(roll), -sin(roll), 0],
                       [sin(roll), cos(roll), 0],
                       [0, 0, 1]])

        R = Rz @ Ry @ Rx

        # Rotate axis
        xaxis = np.dot(R, xaxis) + t
        yaxis = np.dot(R, yaxis) + t
        zaxis = np.dot(R, zaxis) + t
        zaxis1 = np.dot(R, zaxis1) + t

        axis_system = (xaxis, yaxis, zaxis, zaxis1)

        return axis_system

    def project_angles(self, axis_system, anchor_point):
        xaxis, yaxis, zaxis, zaxis1 = axis_system
        xa, ya = anchor_point

        # Gets projected coordinates
        xp_x = (xaxis[0] / xaxis[2] * self.focal_len) + xa
        xp_y = (xaxis[1] / xaxis[2] * self.focal_len) + ya
        xp = (int(xp_x), int(xp_y))

        yp_x = (yaxis[0] / yaxis[2] * self.focal_len) + xa
        yp_y = (yaxis[1] / yaxis[2] * self.focal_len) + ya
        yp = (int(yp_x), int(yp_y))

        zp1_x = (zaxis1[0] / zaxis1[2] * self.focal_len) + xa
        zp1_y = (zaxis1[1] / zaxis1[2] * self.focal_len) + ya
        zp1 = (int(zp1_x), int(zp1_y))

        zp_x = (zaxis[0] / zaxis[2] * self.focal_len) + xa
        zp_y = (zaxis[1] / zaxis[2] * self.focal_len) + ya
        zp = (int(zp_x), int(zp_y))

        projec_coords = (xp, yp, zp, zp1)
        return projec_coords

