import os
import cv2
import sys
import logging as log
import numpy as np
from argparse import ArgumentParser
from src.mouse_controller import MouseController
from src.input_feeder import InputFeeder
from src.visualizer import ShowPreview
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
                             "lm for landmarks, pm for head pose, gm for gaze estimation,"
                             "vo for video only output without models detection output. "
                             "Flags needs to be separated by space.")
    parser.add_argument("-m_prec", "--mouse_precision", type=str, default='high',
                        help="Specify mouse precision (how much the mouse moves): 'high', 'medium', 'low'."
                             "Default is high.")
    parser.add_argument("-m_speed", "--mouse_speed", type=str, default='immediate',
                        help="Specify mouse speed (how many secs before it moves): 'immediate'(0s), 'fast'(0.1s)," 
                             "'medium'(0.5s) and 'slow'(1s). Default is immediate.")
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
    global preview
    args = build_argparser().parse_args()

    # Get input
    input_path = args.input
    flip = False
    if input_path.lower() == 'cam':
        feed = InputFeeder(input_type='cam')
        flip = True
    else:
        if (input_path.endswith('.jpg') or input_path.endswith('.bmp')) and os.path.isfile(input_path):
            feed = InputFeeder('image', input_path)
        elif (input_path.endswith('.avi') or input_path.endswith('.mp4')) and os.path.isfile(input_path):
            feed = InputFeeder('video', input_path)
        else:
            log.error("Specified input file does not exist")
            sys.exit(1)

    feed.load_data()

    # Check models
    models = {'FM': args.face_model,
              'LM': args.landmarks_model,
              'PM': args.pose_model,
              'GM': args.gaze_model}
    for model in models.keys():
        if not os.path.isfile(models[model] + '.xml'):
            log.error("Unable to find specified '" + models[model].split('/')[-1] + "' model")
            sys.exit(1)

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

    # Check preview flags
    if len(args.flags_preview) != 0:
        for flag in args.flags_preview:
            if not flag in ['lm', 'pm', 'fm', 'gm', 'vo']:
                log.error("Flag '" + flag + "' is not a valid preview flag.")
                sys.exit(1)

    # Initialize mouse controller
    mouse = MouseController(precision=args.mouse_precision, speed=args.mouse_speed)
    # out = cv2.VideoWriter("test.mp4", cv2.VideoWriter_fourcc(*"MP4V"), 30,  (1920, 1080), True)
    for frame in feed.next_batch():
        if frame is None:
            break
        key_pressed = cv2.waitKey(60)

        if len(args.flags_preview) != 0:
            preview = ShowPreview(frame, flip)


        # 1: Detect face
        face_coords, crop_face = fm.predict(frame.copy())

        # If face is not detect show a message:
        if len(face_coords) == 0:
            text = "No face detected"
            log.error(text)

        # If face is detect move on with next detections:
        else:
            # Draw face detection if applicable
            if 'fm' in args.flags_preview:
                preview.draw_face_box(face_coords[0], face_coords[1], face_coords[2], face_coords[3])

            # 2a: Detect left and right eye
            eyes_coord, crop_left, crop_right = lm.predict(crop_face.copy())
            # Draw eyes detection if applicable
            if len(eyes_coord) != 0 and 'lm' in args.flags_preview:
                square_size = int(crop_face.shape[1] / 10 + 5)
                # Draw left yes bounding box
                preview.draw_eye_box(eyes_coord[:2], face_coords, square_size)
                # Draw right eye bounding box
                preview.draw_eye_box(eyes_coord[2:4], face_coords, square_size)

            # 2b: Estimate head pose
            angle_poses = pm.predict(crop_face.copy())
            # Draw head pose estimation if applicable
            if len(angle_poses) != 0 and 'pm' in args.flags_preview:
                # Get center of face bounding box
                xcenter = int(face_coords[0] + crop_face.shape[1] / 2)
                ycenter = int(face_coords[1] + crop_face.shape[0] / 2)
                preview.draw_head_pose(
                    yaw=angle_poses[0],
                    pitch=angle_poses[1],
                    roll=angle_poses[2],
                    anchor_point=(xcenter, ycenter)
                )

            # 3: Get Gaze Estimation
            if len(crop_left) != 0 and len(crop_right) != 0:
                gaze = gm.predict(crop_left.copy(), crop_right.copy(), angle_poses.copy())
                # Draw gaze estimation if applicable
                if len(gaze) != 0 and 'gm' in args.flags_preview:
                    gaze_len = (0.4 * crop_face.shape[1], 0.4 * crop_face.shape[0])
                    # Draw left eye gaze
                    preview.draw_eye_gaze(eyes_coord[:2], face_coords, gaze, gaze_len)
                    # Draw right eye gaze
                    preview.draw_eye_gaze(eyes_coord[2:4], face_coords, gaze, gaze_len)

                # Move mouse
                if len(gaze) != 0:
                    if flip:  # if it is a video cam stream, flip pointer direction
                        mouse.move(-gaze[0], gaze[1])
                    else:
                        mouse.move(gaze[0], gaze[1])

        # out.write(preview)
        if len(args.flags_preview) != 0:
            frame = cv2.resize(frame, (600, 600))
            if flip:
                frame = cv2.flip(frame, 1)
            if len(face_coords) == 0:
                text = "No face detected"
                font = cv2.FONT_HERSHEY_COMPLEX
                textsize = cv2.getTextSize(text, font, 1, 2)[0]
                textX, textY = int((600 - textsize[0]) / 2), int((600 + textsize[1]) / 2)
                cv2.putText(frame, text, (textX, textY), font, 1, (200, 10, 10), 1)
                cv2.rectangle(frame, (textX - 10, textY - textsize[1] - 10),
                              (textX + textsize[0] + 10, textY + 10),
                              (145, 50, 255), 2)

            cv2.imshow("Preview", frame)
            cv2.moveWindow("Preview", 70, 70)



        if key_pressed == 27:
            break

    log.error("VideoStream ended...")
    cv2.destroyAllWindows()
    feed.close()


if __name__ == '__main__':
    main()
