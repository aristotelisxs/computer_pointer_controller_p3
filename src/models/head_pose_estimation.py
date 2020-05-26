import os
import cv2
import sys
import numpy as np

from openvino.inference_engine import IECore


class HeadPoseEstimationModel:
    '''
    Class for the Head Pose Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        self.net = None
        self.plugin = None
        self.exec_net = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None

        self.device = device
        self.model_name = model_name
        # assert(os.path.isfile(model_name))

        self.extensions = extensions
        self.model_weights = "." + self.model_name.split(".")[1] + '.bin'

    def load_model(self):
        """
        Loads the head pose estimation model to the device specified by the user.
        Any plugins are loaded here.
        """
        self.plugin = IECore()
        self.net = self.plugin.read_network(model=self.model_name, weights=self.model_weights)

        self.check_model()
        self.exec_net = self.plugin.load_network(network=self.net, device_name=self.device, num_requests=1)
        
        self.input_name = next(iter(self.net.inputs))
        self.input_shape = self.net.inputs[self.input_name].shape
        self.output_names = [i for i in self.net.outputs.keys()]
        
    def predict(self, img):
        """
        Run predictions on the input image to estimate the head pose of the face given from the frame/image.
        :param img: (cv2.image) The current frame/image to be pre-processed
        """
        processed_img = self.preprocess_input(img.copy())
        outputs = self.exec_net.infer({
            self.input_name: processed_img
        })

        prediction = self.preprocess_output(outputs)

        return prediction

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

    def preprocess_input(self, img):
        """
        Pre-process the image to be (no_of_batches, no_of_channels, width, height)
        :param img: (cv2.image) The current frame/image to be pre-processed
        """
        img_resized = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))
        processed_img = np.transpose(np.expand_dims(img_resized, axis=0), (0, 3, 1, 2))

        return processed_img

    def preprocess_output(self, outputs):
        """
        Preprocess outputs to be fed to the next model (gaze estimation model specifically).
        :param outputs: (dict of lists) Set of outputs corresponding to the vectors yaw, pitch and roll that describe
        the head pose. Assume only a single face in the image.
        """
        pp_out = []
        pp_out.append(outputs['angle_y_fc'].tolist()[0][0])
        pp_out.append(outputs['angle_p_fc'].tolist()[0][0])
        pp_out.append(outputs['angle_r_fc'].tolist()[0][0])

        return pp_out
