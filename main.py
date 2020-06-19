import os
import cv2
import logging as log
from argparse import ArgumentParser
from src.face_detection import FaceDetectionModel
from src.facial_landmarks_detection import LandmarksDetectionModel
from src.input_feeder import InputFeeder


def build_argparser():
    '''
    Parse command line arguments.

    :return: command line arguments
    '''
    parser = ArgumentParser()
    parser.add_argument("-fm", "--face_model", required=True, type=str,
                        help="Path to folder with a pre-trained 'Face Detection Model'. E.g. <path_dir>/<model_name>")
    # parser.add_argument("-pm", "--pose_model", required=True, type=str,
    #                     help="Path to folder with a pre-trained 'Head Pose Detection Model'. . E.g. <path_dir>/<model_name>")
    parser.add_argument("-lm", "--landmarks_model", required=True, type=str,
                          help="Path to folder with a pre-trained 'Facial Landmarks Detection Model'. . E.g. <path_dir>/<model_name>")
    # parser.add_argument("-gm", "--gaze_model", required=True, type=str,
    #                     help="Path to folder with a pre-trained 'Gaze Estimation Model'. . E.g. <path_dir>/<model_name>")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to video file or image. Enter 'cam' for webcam stream.")
    parser.add_argument("-l", "--extensions", type=str, default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl")
    parser.add_argument("-f", "--flags_preview", nargs="+", default=[],
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
              'LM': args.landmarks_model}
    # 'PM': args.pose_model,
    # 'GM': args.gaze_model}
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

    fm.load_model()
    lm.load_model()

    for frame in feed.next_batch():
        if frame is None:
            break
        key_pressed = cv2.waitKey(60)
        preview = frame.copy()
        prev_w, prev_h = preview.shape[1], preview.shape[0]

        # 1: Detect face
        face_coords, crop_face = fm.predict(frame.copy())
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
            eyes_coords,crop_left, crop_right = lm.predict(crop_face.copy())
            # Draw eyes detection if applicable
            if 'lm' in args.flags_preview:
                square_size = int(crop_face.shape[0]/10)
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


        cv2.imshow("Preview", preview)

        if key_pressed == 27:
            break

    feed.close()


if __name__ == '__main__':
    main()
