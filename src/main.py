import os
import sys
import cv2
import time
import logging

from stat import *
from sys import platform
from models.face_detection import FaceDetectionModel
from models.facial_landmarks_detection import FacialLandmarksDetectionModel
from models.gaze_estimation import GazeEstimationModel
from models.head_pose_estimation import HeadPoseEstimationModel

from argparse import ArgumentParser, ArgumentTypeError
from input_feeder import InputFeeder

mc_lib_loaded = False
try:
    from mouse_controller import MouseController
    mc_lib_loaded = True
except:
    # This is probably used for benchmarking. If not, raise an exception later
    pass


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def prep_models(cla):
    fdm = FaceDetectionModel(cla.face_detection_model, cla.device, cla.cpu_extension)
    fdm.load_model()

    fldm = FacialLandmarksDetectionModel(cla.facial_landmark_model, cla.device, cla.cpu_extension)
    fldm.load_model()

    hpem = HeadPoseEstimationModel(cla.head_pose_model, cla.device, cla.cpu_extension)
    hpem.load_model()

    gem = GazeEstimationModel(cla.gaze_estimation_model, cla.device, cla.cpu_extension)
    gem.load_model()

    return fdm, fldm, hpem, gem


def visualize_pipeline_results(frame, cropped_face, eye_coords, left_eye, right_eye, hp_out, gaze_vector,
                               preview_flags):
    preview_frame = frame.copy()
    if 'fd' in preview_flags:
        preview_frame = cropped_face
    if 'fld' in preview_flags:
        cv2.rectangle(cropped_face, (eye_coords[0][0] - 10, eye_coords[0][1] - 10),
                      (eye_coords[0][2] + 10, eye_coords[0][3] + 10), (0, 255, 0), 3)
        cv2.rectangle(cropped_face, (eye_coords[1][0] - 10, eye_coords[1][1] - 10),
                      (eye_coords[1][2] + 10, eye_coords[1][3] + 10), (0, 255, 0), 3)
    if 'hp' in preview_flags:
        cv2.putText(preview_frame,
                    "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(hp_out[0], hp_out[1],
                                                                                  hp_out[2]), (10, 20),
                    cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 255, 0), 1)
    if 'ge' in preview_flags:
        x, y, w = int(gaze_vector[0] * 12), int(gaze_vector[1] * 12), 160
        le = cv2.line(left_eye.copy(), (x - w, y - w), (x + w, y + w), (255, 0, 255), 2)
        cv2.line(le, (x - w, y + w), (x + w, y - w), (255, 0, 255), 2)
        re = cv2.line(right_eye.copy(), (x - w, y - w), (x + w, y + w), (255, 0, 255), 2)
        cv2.line(re, (x - w, y + w), (x + w, y - w), (255, 0, 255), 2)
        cropped_face[eye_coords[0][1]:eye_coords[0][3], eye_coords[0][0]:eye_coords[0][2]] = le
        cropped_face[eye_coords[1][1]:eye_coords[1][3], eye_coords[1][0]:eye_coords[1][2]] = re

    return preview_frame


def handle_input_feed(logger, preview_flags, fdm, fldm, hpem, gem, mc, in_feeder, frame_out_rate, codec, output_path):
    """
    Feeds inputs to models, moving the mouse cursor based on the final gaze estimation.
    :param logger: (logging.logger) Logger object
    :param preview_flags: (list) List of visualisations the user would like to see
    :param fdm: (openvino model) Face detection loaded model
    :param fldm: (openvino model) Facial landmarks detection loaded model
    :param hpem: (openvino model) Head pose estimation loaded model
    :param gem: (openvino model) Gaze estimation loaded model
    :param mc: Mouse controller that will move the mouse pointer depending on the gaze estimation model's outputs
    :param in_feeder: (input_feeder.InputFeeder) Feeds input from an image, webcam, or video to our face detection model
    (gateway model to the pipeline)
    :param frame_out_rate: (int) After how many frames a processed frame will be displayed to each of the visualisations
    requested by the user?
    :param codec: Depending on the platform this is run on, OpenCV requires a codec to be specified. Supply it here.
    :param output_path: Where to save the video file that will be produced if no live visualisation is requested
    :return: None
    """
    frame_count = 0
    out = cv2.VideoWriter(os.path.join(output_path, 'out.mp4'), codec, 30, (650, 650))
    start_inference_time = time.time()

    for ret, frame in in_feeder.next_batch():
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_out_rate == 0 and mc_lib_loaded:
            cv2.imshow('video', cv2.resize(frame, (650, 650)))

        key = cv2.waitKey(60)

        # -- Gaze estimation pipeline -- #
        cropped_face, face_coords = fdm.predict(frame.copy(), args.prob_threshold)

        if type(cropped_face) == int:
            logger.error("Unable to detect a face.")
            if key == 27:
                break
            continue

        hp_out = hpem.predict(cropped_face.copy())
        left_eye, right_eye, eye_coords = fldm.predict(cropped_face.copy())
        new_mouse_coord, gaze_vector = gem.predict(left_eye, right_eye, hp_out)
        # -- End Gaze estimation pipeline -- #

        preview_frame = visualize_pipeline_results(frame, cropped_face, eye_coords, left_eye, right_eye, hp_out,
                                                   gaze_vector, preview_flags)

        if not len(preview_flags) == 0 and mc_lib_loaded:
            cv2.imshow("visualization", cv2.resize(preview_frame, (650, 650)))
        elif out is not None:
            out.write(cv2.resize(preview_frame, (650, 650)))

        if frame_count % frame_out_rate == 0 and mc is not None:
            mc.move(new_mouse_coord[0], new_mouse_coord[1])
        if key == 27:
            break

    total_time = time.time() - start_inference_time
    total_inference_time = round(total_time, 1)
    fps = frame_count / total_inference_time

    return fps, total_inference_time, total_time


def walktree(top, callback):
    '''
        recursively descend the directory tree rooted at top,
        calling the callback function for each regular file
    '''
    for f in os.listdir(top):
        try:
            pathname = os.path.join(top, f)
            mode = os.stat(pathname)[ST_MODE]
            if S_ISDIR(mode):
                # It's a directory, recurse into it
                walktree(pathname, callback)
            elif S_ISREG(mode):
                # It's a file, call the callback function
                callback(pathname)
            else:
                # Unknown file type, print a message
                print('Skipping %s' % pathname)
        except:
            pass


def visit_file(file):
    print('Visiting ', file)


def start_pipeline(cla, codec):
    """
    Initializes feeds inputs to models, moving the mouse cursor based on the final gaze estimation.
    :param cla: Command line arguments for configuring the pipeline.
    :param codec: Depending on the platform this is run on, OpenCV requires a codec to be specified. Supply it here.
    :return: None
    """
    preview_flags = cla.preview_flags

    logger = logging.getLogger()
    input_file_path = cla.input

    if input_file_path.lower() == "cam":
        in_feeder = InputFeeder("cam")
    elif not os.path.isfile(input_file_path):
        # top = os.path.dirname(os.path.realpath(__file__))
        # walktree(top, visit_file)
        logger.error("Cannot locate video file provided. Exiting..")
        sys.exit(0)
    else:
        in_feeder = InputFeeder("video", input_file_path)

    start_model_load_time = time.time()
    fdm, fldm, hpem, gem = prep_models(cla)
    total_model_load_time = time.time() - start_model_load_time

    mc = None
    if not cla.is_benchmark:
        mc = MouseController('medium', 'fast')

    in_feeder.load_data()

    fps, total_inference_time, total_time = handle_input_feed(logger, preview_flags, fdm, fldm, hpem, gem, mc, in_feeder,
                                                  cla.frame_out_rate, codec, cla.output_path)

    with open(os.path.join(cla.output_path, 'stats.txt'), 'w') as f:
        f.write("Total inference time, " + str(total_inference_time) + '\n')
        f.write("FPS, " + str(fps) + '\n')
        f.write("Total model load time, " + str(total_model_load_time) + '\n')
        f.write("Total time, " + str(total_time) + '\n')

    logger.error("Video stream ended...")
    cv2.destroyAllWindows()
    in_feeder.close()


if __name__ == '__main__':
    parser = ArgumentParser()

    # --- Models in the pipeline --- #
    parser.add_argument("-f", "--face_detection_model", required=True, type=str,
                        help="Path to the .xml file of Face Detection model.")
    parser.add_argument("-fl", "--facial_landmark_model", required=True, type=str,
                        help="Path to the to .xml file of Facial Landmark Detection model.")
    parser.add_argument("-hp", "--head_pose_model", required=True, type=str,
                        help="Path to the to .xml file of Head Pose Estimation model.")
    parser.add_argument("-g", "--gaze_estimation_model", required=True, type=str,
                        help="Path to the to .xml file of Gaze Estimation model.")

    # --- Pipeline input arguments --- #
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to the to video file. Input 'cam' to use webcam.")
    parser.add_argument("-fout", "--frame_out_rate", required=False, type=int, default=5,
                        help="After how many frames should video be displayed (given that flags are specified"
                        )
    parser.add_argument("-flags", "--preview_flags", required=False, nargs='+',
                        default=list(),
                        help="Use the following, separated by spaces, to get video outputs for:\n"
                             "fd for Face Detection, \n"
                             "fld for Facial Landmark Detection\n"
                             "hp for Head Pose Estimation and,\n"
                             "ge for Gaze Estimation.\n")
    parser.add_argument("-prob", "--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="The confidence the model needs to demonstrate to accept face detections.")

    # --- Device related --- #
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on among the following:\n"
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Make sure device specified is plugged in "
                             "(CPU by default)")
    parser.add_argument("-bench", "--is_benchmark", type=str2bool, nargs='?', required=False, default=True,
                        help="Specify whether this will be run for benchmarking to disable mouse controlling.")

    # --- Miscellaneous --- #
    parser.add_argument("-out", "--output_path", required=False, type=str, default=".",
                        help="Path to save statistics and output video files")

    args = parser.parse_args()

    if not mc_lib_loaded:
        # Probably within the Intel DevCloud instead of a local machine. Use this codec.
        codec = cv2.VideoWriter_fourcc(*'avc1')
    elif platform == "linux" or platform == "linux2":
        codec = 0x00000021
    elif platform == "darwin":
        codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    else:
        print("Unsupported OS.")
        sys.exit(0)

    assert((not args.is_benchmark and mc_lib_loaded) or args.is_benchmark)
    start_pipeline(args, codec)
