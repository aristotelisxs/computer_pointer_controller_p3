import os
import cv2
import sys
import numpy as np

from openvino.inference_engine import IECore


class FaceDetectionModel:
    """
    Class for the Face Detection Model.
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
        
        self.input_name = next(iter(self.net.inputs))
        self.input_shape = self.net.inputs[self.input_name].shape
        self.output_names = next(iter(self.net.outputs))
        self.output_shape = self.net.outputs[self.output_names].shape
        
    def predict(self, img, prob_threshold):
        """
        Run predictions on the input image to estimate where the face in the given frame/image is.
        :param img: (cv2.image) The current frame/image to be pre-processed
        :param prob_threshold: (float) The confidence the model needs to demonstrate to accept a prediction
        """
        img_processed = self.preprocess_input(img.copy())
        outputs = self.exec_net.infer({
            self.input_name: img_processed
        })
        coords = self.preprocess_output(outputs, prob_threshold)
        if len(coords) == 0:
            return 0, 0

        coords = coords[0]  # Account for only a single face in the image.
        coords = coords * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        coords = coords.astype(np.int32)
        
        cropped_face = img[coords[1]:coords[3], coords[0]:coords[2]]

        return cropped_face, coords

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
                    print("Extension provided still does not support all layers found. Exiting..")
                    sys.exit(0)
                print("Extension supports all layers found")
            else:
                print("Path to the cpu extension is not provided. Exiting..")
                sys.exit(0)

    def preprocess_input(self, img):
        """
        Pre-process the image to be (no_of_batches, no_of_channels, width, height)
        :param img: (cv2.image) The current frame/image to be pre-processed
        """
        img_resized = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))
        processed_img = np.transpose(np.expand_dims(img_resized, axis=0), (0, 3, 1, 2))

        return processed_img

    def preprocess_output(self, outputs, prob_threshold):
        """
        Preprocess outputs to be fed to the next model (the facial landmarks detection and head pose estimation
        specifically).
        :param outputs: (list of lists) Set of outputs corresponding to coordinates of bounding boxes for any
        predictions that were made by the model.
        :param prob_threshold: (float) The confidence the model needs to demonstrate to accept a prediction
        """
        coords = list()
        outs = outputs[self.output_names][0][0]

        for out in outs:
            conf = out[2]
            if conf >= prob_threshold:
                x_min = out[3]
                y_min = out[4]
                x_max = out[5]
                y_max = out[6]
                coords.append([x_min, y_min, x_max, y_max])

        return coords
