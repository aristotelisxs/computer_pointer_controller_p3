import os
import cv2
import sys
import numpy as np

from openvino.inference_engine import IECore


class FacialLandmarksDetectionModel:
    '''
    Class for the Facial Landmarks Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        self.net = None
        self.plugin = None
        self.exec_net = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.output_shape = None

        self.device = device
        self.position_offset = 10
        self.model_name = model_name
        # assert(os.path.isfile(model_name))

        self.extensions = extensions
        self.model_weights = "." + self.model_name.split(".")[1] + '.bin'

    def load_model(self):
        """
        Loads the facial landmarks detection model to the device specified by the user.
        Any plugins are loaded here.
        """
        self.plugin = IECore()
        self.net = self.plugin.read_network(model=self.model_name, weights=self.model_weights)

        self.check_model()
        self.exec_net = self.plugin.load_network(network=self.net, device_name=self.device, num_requests=1)
        
        self.input_name = next(iter(self.net.inputs))
        self.input_shape = self.net.inputs[self.input_name].shape
        self.output_names = next(iter(self.net.outputs))
        self.output_shape = self.net.outputs[self.output_names].shape
        
    def predict(self, img):
        """
        Run predictions on the input image to estimate which are the facial landmarks in the face given from a
        frame/image.
        :param img: (cv2.image) The current frame/image to be pre-processed
        """
        img_processed = self.preprocess_input(img.copy())
        outputs = self.exec_net.infer({
            self.input_name:img_processed
        })

        coords = self.preprocess_output(outputs)
        # rectangles with shape (w, h, w, h)
        coords = coords * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        coords = coords.astype(np.int32)

        l_eye_min_x = coords[0] - self.position_offset
        l_eye_min_y = coords[1] - self.position_offset
        l_eye_max_x = coords[0] + self.position_offset
        l_eye_max_y = coords[1] + self.position_offset
        
        r_eye_min_x = coords[2] - self.position_offset
        r_eye_min_y = coords[3] - self.position_offset
        r_eye_max_x = coords[2] + self.position_offset
        r_eye_max_y = coords[3] + self.position_offset

        l_eye = img[l_eye_min_y:l_eye_max_y, l_eye_min_x:l_eye_max_x]
        r_eye = img[r_eye_min_y:r_eye_max_y, r_eye_min_x:r_eye_max_x]

        eyes_pos = [
            [l_eye_min_x, l_eye_min_y, l_eye_max_x, l_eye_max_y],
            [r_eye_min_x, r_eye_min_y, r_eye_max_x, r_eye_max_y]
        ]

        return l_eye, r_eye, eyes_pos
        
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
                    raise Exception("Extension provided still does not support all layers found. Exiting")
                print("Extension supports all layers found")
            else:
                raise Exception("Path to the cpu extension is not provided. Exiting")

    def preprocess_input(self, img):
        """
        Pre-process the image to be (no_of_batches, no_of_channels, width, height)
        :param img: (cv2.image) The current frame/image to be pre-processed
        """
        # Inputs to the facial landmarks detection model need to be in RGB.
        image_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(image_cvt, (self.input_shape[3], self.input_shape[2]))
        processed_img = np.transpose(np.expand_dims(img_resized, axis=0), (0, 3, 1, 2))

        return processed_img

    def preprocess_output(self, outputs):
        """
        Preprocess outputs to be fed to the next model (gaze estimation model specifically).
        :param outputs: (dict of lists) Set of outputs corresponding to coordinates of bounding boxes for any
        facial landmarks detected (the left and right eyes here).
        """
        outs = outputs[self.output_names][0]
        l_eye_x = outs[0].tolist()[0][0]
        l_eye_y = outs[1].tolist()[0][0]
        r_eye_x = outs[2].tolist()[0][0]
        r_eye_y = outs[3].tolist()[0][0]
        
        return (l_eye_x, l_eye_y, r_eye_x, r_eye_y)
