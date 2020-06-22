import os
import cv2
import logging as log
import numpy as np
from math import cos, sin, pi
from argparse import ArgumentParser
from src.input_feeder import InputFeeder
from src.face_detection import FaceDetectionModel
from src.gaze_estimation import GazeEstimationModel
from src.head_pose_estimation import HeadPoseEstimationModel
from src.facial_landmarks_detection import LandmarksDetectionModel




def build_argparser():
    '''
    Parse command line arguments.

    :return: command line arguments
    '''
    parser = ArgumentParser()
    parser.add_argument("-fm", "--face_model", required=True, type=str,
                        help="Path to folder with a pre-trained 'Face Detection Model'. E.g. <path_dir>/<model_name>")
    parser.add_argument("-pm", "--pose_model", required=True, type=str,
                         help="Path to folder with a pre-trained 'Head Pose Detection Model'. . E.g. <path_dir>/<model_name>")
    parser.add_argument("-lm", "--landmarks_model", required=True, type=str,
                          help="Path to folder with a pre-trained 'Facial Landmarks Detection Model'. . E.g. <path_dir>/<model_name>")
    parser.add_argument("-gm", "--gaze_model", required=True, type=str,
                         help="Path to folder with a pre-trained 'Gaze Estimation Model'. . E.g. <path_dir>/<model_name>")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or image. Enter 'cam' for webcam stream.")
    parser.add_argument("-l", "--extensions", type=str, default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl")
    parser.add_argument("-prev", "--flags_preview", nargs="+", default=[],
                        help="Show models detection outputs. Add 'fm' for face detection,"
                             "lm for landmarks, pm for head pose, gm for gaze estimation")
    parser.add_argument("-prob", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detection filtering (0.5 by default)")
    parser.add_argument("-d", "--device", type=str, default='CPU',
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable")
    parser.add_argument("-a", "--async_mode", type=str, default=True,
                        help="Specify the inference type. True for asynchronous, false for synchronous.")

    return parser


def main():
    # Grab command line arguments
    args = build_argparser().parse_args()

    # Get input
    input_path = args.input
    if input_path.lower() == 'cam':
        feed = InputFeeder(input_type='cam')
    else:
        if ((input_path.endswith('.jpg') or input_path.endswith('.bmp')) and os.path.isfile(input_path)):
            feed = InputFeeder('image', input_path)
        elif ((input_path.endswith('.avi') or input_path.endswith('.mp4')) and os.path.isfile(input_path)):
            feed = InputFeeder('video', input_path)
        else:
            log.error("Specified input file does not exist")
            exit(1)

    feed.load_data()

    # Check models
    models = {'FM': args.face_model,
              'LM': args.landmarks_model,
              'PM': args.pose_model,
              'GM': args.gaze_model}
    for model in models.keys():
        if not os.path.isfile(models[model] + '.xml'):
            log.error("Unable to find specified '" + models[model].split('/')[-1] + "' model")
            exit(1)

    # Load models
    fm = FaceDetectionModel(model_name=models['FM'],
                            prob_threshold=args.prob_threshold,
                            device=args.device,
                            extensions=args.extensions,
                            async_infer=args.async_mode)
    lm = LandmarksDetectionModel(model_name=models['LM'],
                            device=args.device,
                            extensions=args.extensions,
                            async_infer=args.async_mode)
    pm = HeadPoseEstimationModel(model_name=models['PM'],
                                 device=args.device,
                                 extensions=args.extensions,
                                 async_infer=args.async_mode)
    gm = GazeEstimationModel(model_name=models['GM'],
                                 device=args.device,
                                 extensions=args.extensions,
                                 async_infer=args.async_mode)

    fm.load_model()
    lm.load_model()
    pm.load_model()
    gm.load_model()

    for preview in feed.next_batch():
        if preview is None:
            break
        key_pressed = cv2.waitKey(60)
        preview = preview.copy()
        prev_w, prev_h = preview.shape[1], preview.shape[0]

        # 1: Detect face
        face_coords, crop_face = fm.predict(preview.copy())
        # If face is not detect show a message:
        if len(face_coords) == 0:
            text = "No face detected"
            font = cv2.FONT_HERSHEY_COMPLEX
            textsize = cv2.getTextSize(text, font, 1, 2)[0]
            textX, textY = int((prev_w - textsize[0])/2), int((prev_h + textsize[1])/2)
            cv2.putText(preview, text, (textX, textY), font, 1, (200, 10, 10), 1)
            cv2.rectangle(preview, (textX-10, textY-textsize[1]-10), (textX+textsize[0]+10, textY+10), (145, 50, 255), 2)

        # If face is detect move on with next detections:
        else:
            # Draw face detection if applicable
            if 'fm' in args.flags_preview:
                cv2.rectangle(preview, (face_coords[0], face_coords[1]), (face_coords[2], face_coords[3]),
                              (145, 50, 255), 2)

            # 2a: Detect left and right eye
            eyes_coords, crop_left, crop_right = lm.predict(crop_face.copy())
            # Draw eyes detection if applicable
            if 'lm' in args.flags_preview:
                square_size = int(crop_face.shape[1]/10 + 5 )
                # Draw left yes bounding box
                xl_min, yl_min = eyes_coords[0] + face_coords[0] - square_size, eyes_coords[1] + face_coords[1] - square_size
                xl_max, yl_max = eyes_coords[0] + face_coords[0] + square_size, eyes_coords[1] + face_coords[1] + square_size
                cv2.rectangle(preview, (xl_min, yl_min), (xl_max, yl_max), (200, 10, 10), 2)
                # Draw right eye bounding box
                xr_min, yr_min = eyes_coords[2] + face_coords[0] - square_size, eyes_coords[3] + face_coords[
                    1] - square_size
                xr_max, yr_max = eyes_coords[2] + face_coords[0] + square_size, eyes_coords[3] + face_coords[
                    1] + square_size
                cv2.rectangle(preview, (xr_min, yr_min), (xr_max, yr_max), (200, 10, 10), 2)

            # 2b: Estimate head pose
            angle_poses = pm.predict(crop_face.copy())
            # Draw head pose estimation if applicable
            if 'pm' in args.flags_preview:
                # Transform Tait-Bryan angles to radians
                pitch = angle_poses[1] * pi/180
                yaw = angle_poses[0] * pi/180
                roll = angle_poses[2] * pi/180


                xcenter = int(face_coords[0] + crop_face.shape[1]/2)
                ycenter = int(face_coords[1] + crop_face.shape[0]/2)


                # Create axis where to project angles
                # ref1: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
                # ref2: https://github.com/opencv/open_model_zoo/blob/master/demos/interactive_face_detection_demo/visualizer.cpp
                scale = 50
                focal_len = 950.0
                xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
                yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
                zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
                zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)


                # Translation matrix
                t = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
                t[2] = focal_len

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

                # Gets projected coordinates & draw lines
                xp2 = (xaxis[0] / xaxis[2] * focal_len) + xcenter
                yp2 = (xaxis[1] / xaxis[2] * focal_len) + ycenter
                p2 = (int(xp2), int(yp2))
                cv2.line(preview, (xcenter, ycenter), p2, (0, 0, 255), 2) # x-axis

                xp2 = (yaxis[0] / yaxis[2] * focal_len) + xcenter
                yp2 = (yaxis[1] / yaxis[2] * focal_len) + ycenter
                p2 = (int(xp2), int(yp2))
                cv2.line(preview, (xcenter, ycenter), p2, (0, 255, 0), 2) # y-axis

                xp1 = (zaxis1[0] / zaxis1[2]* focal_len) + xcenter
                yp1 = (zaxis1[1] / zaxis1[2]* focal_len) + ycenter
                p1 = (int(xp1), int(yp1))
                xp2 = (zaxis[0] / zaxis[2]* focal_len) + xcenter
                yp2 = (zaxis[1] / zaxis[2]* focal_len) + ycenter
                p2 = (int(xp2), int(yp2))
                cv2.line(preview, p1, p2, (255, 0, 0), 2) # z-axis
                cv2.circle(preview, p2, 3, (255, 0, 0), 2)

            # 3: Get Gaze Estimation
            if len(crop_left) !=0 and len(crop_right) != 0:
                gaze = gm.predict(crop_left.copy(), crop_right.copy(), angle_poses.copy())
                # Draw gaze estimation if applicable
                if len(gaze) != 0  and 'gm' in args.flags_preview:
                 left_xcenter = int(eyes_coords[0] + face_coords[0])
                 left_ycenter = int(eyes_coords[1] + face_coords[1])
                 right_xcenter = int(eyes_coords[2] + face_coords[0])
                 right_ycenter = int(eyes_coords[3] + face_coords[1])

                 gaze_lx = (gaze[0] * 0.4 * crop_face.shape[1]) + left_xcenter
                 gaze_ly = (-gaze[1] * 0.4 * crop_face.shape[0]) + left_ycenter
                 gp_l = (int(gaze_lx), int(gaze_ly))
                 gaze_rx = (gaze[0] * 0.4 * crop_face.shape[1]) + right_xcenter
                 gaze_ry = (-gaze[1] * 0.4 * crop_face.shape[0]) + right_ycenter
                 gp_r = (int(gaze_rx), int(gaze_ry))
                 cv2.arrowedLine(preview, (left_xcenter, left_ycenter), gp_l, (230, 216, 173), 2)
                 cv2.arrowedLine(preview, (right_xcenter, right_ycenter), gp_r, (230, 216, 173), 2)



        cv2.imshow("Preview", preview)

        if key_pressed == 27:
            break

    feed.close()


if __name__ == '__main__':
    main()
