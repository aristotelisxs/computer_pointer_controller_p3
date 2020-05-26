import os
import cv2
import sys
import math
import numpy as np

from openvino.inference_engine import IECore


class GazeEstimationModel:
    """
    Class for the Gaze Detection Model.
    """
    def __init__(self, model_name, device='CPU', extensions=None):
        self.net = None
        self.plugin = None
        self.exec_net = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.output_shape = None

        self.device = device
        self.model_name = model_name
        # assert(os.path.isfile(model_name))

        self.extensions = extensions
        self.model_weights = "." + self.model_name.split(".")[1] + '.bin'

    def load_model(self):
        """
        Loads the face detection model to the device specified by the user.
        Any plugins are loaded here.
        """
        self.plugin = IECore()
        self.net = self.plugin.read_network(model=self.model_name, weights=self.model_weights)

        self.check_model()
        self.exec_net = self.plugin.load_network(network=self.net, device_name=self.device, num_requests=1)
        
        self.input_name = [i for i in self.net.inputs.keys()]
        self.input_shape = self.net.inputs[self.input_name[1]].shape
        self.output_names = [i for i in self.net.outputs.keys()]

    def predict(self, l_eye_image, r_eye_image, head_pose_angles):
        """
        Run predictions on the input image to estimate where the eyes are gazing in the face given from frame/image.
        :param l_eye_image: (cv2.image) The current frame/image to be pre-processed
        :param r_eye_image: (cv2.image) The current frame/image to be pre-processed
        :param head_pose_angles: (list-like) Angle that the face in the given frame/image is currently positioned at.
        """
        l_eye_pp_img, r_eye_pp_img = self.preprocess_input(l_eye_image.copy(), r_eye_image.copy())
        outputs = self.exec_net.infer({
            'head_pose_angles': head_pose_angles,
            'left_eye_image': l_eye_pp_img,
            'right_eye_image': r_eye_pp_img
        })

        out_mouse_coord, gaze_vector = self.preprocess_output(outputs, head_pose_angles)

        return out_mouse_coord, gaze_vector

    def check_model(self):
        """
        Check whether any layers are not supported and if we can account for un-supported layers through extensions
        provided. Exit if all fail.
        """
        supported_layers = self.plugin.query_network(network=self.net, device_name=self.device)
        unsupported_layers = [layer for layer in self.net.layers.keys() if layer not in supported_layers]

        if len(unsupported_layers) > 0 and self.device == 'CPU':
            print("unsupported layers found:{}".format(unsupported_layers))
            if self.extensions is not None:
                print("Adding cpu_extension")
                self.plugin.add_extension(self.extensions, self.device)
                supported_layers = self.plugin.query_network(network=self.net, device_name=self.device)
                unsupported_layers = [layer for layer in self.net.layers.keys() if layer not in supported_layers]

                if len(unsupported_layers) > 0:
                    print("Extension provided still does not support all layers found. Exiting")
                    sys.exit(0)
                print("Extension supports all layers found")
            else:
                print("Path to the cpu extension is not provided. Exiting")
                sys.exit(0)

    def preprocess_input(self, l_eye_image, r_eye_image):
        """
        Pre-process the images to be (no_of_batches, no_of_channels, width, height)
        :param l_eye_image: (cv2.image) The current frame/image to be pre-processed
        :param r_eye_image: (cv2.image) The current frame/image to be pre-processed
        """
        l_eye_image_resized = cv2.resize(l_eye_image, (self.input_shape[3], self.input_shape[2]))
        r_eye_image_resized = cv2.resize(r_eye_image, (self.input_shape[3], self.input_shape[2]))
        l_eye_processed_img = np.transpose(np.expand_dims(l_eye_image_resized, axis=0), (0, 3, 1, 2))
        r_eye_processed_img = np.transpose(np.expand_dims(r_eye_image_resized, axis=0), (0, 3, 1, 2))

        return l_eye_processed_img, r_eye_processed_img

    def preprocess_output(self, outputs, head_pose_angles):
        """
        Preprocess outputs to move the last stage of the pipeline (moving the mouse pointer). Here, we estimate gaze
        using the actual gaze direction and estimated gaze direction. This can be found at the intersection angle \theta
        between the two. Note that we consider zero distance from the monitor screen. More details can be found here:
        https://www.mdpi.com/2076-3417/6/6/174/pdf
        :param outputs: (dict) Set of outputs corresponding to gaze estimations that were made by the model
        :param head_pose_angles: (list-like) Angle that the face in the given frame/image is currently positioned at.
        """
        gaze_vector = outputs[self.output_names[0]].tolist()[0]

        roll_value = head_pose_angles[2]  # shape: [1, 1] - Estimated roll (in degrees)
        cos_value = math.cos(roll_value * math.pi / 180.0)
        sin_value = math.sin(roll_value * math.pi / 180.0)
        
        new_x = gaze_vector[0] * cos_value + gaze_vector[1] * sin_value
        new_y = -gaze_vector[0] * sin_value + gaze_vector[1] * cos_value

        return (new_x, new_y), gaze_vector
