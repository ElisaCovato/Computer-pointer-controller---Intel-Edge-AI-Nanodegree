import logging as log
import cv2
import sys
import numpy as np
from openvino.inference_engine import IECore

class GazeEstimationModel:
    '''
    Class for the Gaze Estimation Model.

    Load and configure inference plugins for the specified target devices,
    and performs either synchronous or asynchronous modes for the
    specified infer requests.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, async_infer=True):
        '''
        Set instance variables.
        '''
        self.plugin = None
        self.network = None
        self.exec_network = None
        self.infer_request_handle = None

        self.input_blob = None
        self.input_shape = None
        self.output_blob = None
        self.output_shape = None

        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.async_infer = async_infer



    def load_model(self):
        '''
        This method is for loading the model (in IR format) to the device specified by the user.
        Default device is CPU.
        '''

        # Get model
        model_structure = self.model_name + '.xml'
        model_weights = self.model_name + '.bin'

        # Initialize the plugin - load the inference engine API
        self.plugin = IECore()

        # Add a CPU extension, if applicable
        if self.extensions and 'CPU' in self.device:
            self.plugin.add_extension(self.extensions, self.device)

        # Read the IR as IENetwork
        try:
            self.network = self.plugin.read_network(model = model_structure, weights = model_weights)
        except:
            raise ValueError("Could not initialise the network. Have you entered the correct model path?")

        # Check if model and plugin are supported
        self.check_model()

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(network = self.network, device_name = self.device, num_requests = 1)

        # Get the input and output layers
        self.input_blob = [key for key in self.network.inputs.keys()]
        self.input_shape = []
        for input in self.input_blob:
            self.input_shape.append(self.network.inputs[input].shape)
        self.output_blob = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_blob].shape
        return

    def predict(self, left_eye, right_eye, angles):
        '''
        This method is meant for running predictions on the input image.
        '''
        # Create input image to feed into the network
        angles = np.array(angles)
        angles = angles.reshape(1, *angles.shape)
        if np.all(np.array(left_eye.shape)) and np.all(np.array(right_eye.shape)):
            net_input = {self.input_blob[0]: angles,
                         self.input_blob[1]: self.preprocess_input(left_eye),
                         self.input_blob[2]: self.preprocess_input(right_eye)}

            # Start inference. Infer mode (async/sync) is input by user
            if self.async_infer:
                self.infer_request_handle = self.exec_network.start_async(request_id = 0, inputs = net_input)
            else:
                self.infer_request_handle = self.exec_network.infer(inputs = net_input)

            # # Wait for the result of the inference
            if self.exec_network.requests[0].wait(-1) == 0:
                # Get result of the inference request
                outputs = self.infer_request_handle.outputs[self.output_blob]
                gaze = self.preprocess_output(outputs)
        else:
            gaze = []

        return gaze

    def check_model(self):
        '''
        This method check whether the model (along with the plugin) is support on the CPU device.
        If anything is missing (such as a CPU extension), let the user know and exit the programm.
        '''

        supported_layers = self.plugin.query_network(network=self.network, device_name='CPU')
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]

        if len(unsupported_layers) != 0:
            log.error("Unsupported layers found: {}".format(unsupported_layers))
            if self.extensions:
                log.error("The extensions specified do not support some layers. Please specify a new extension.")
            else:
                log.error("Please try to specify an extension library path by using the --extensions command line argument.")
            sys.exit(1)
        return

    def preprocess_input(self, image):
        '''
        Method to process inputs before feeding them into the model for inference.
        '''
        image = cv2.resize(image, (self.input_shape[1][3], self.input_shape[1][2]))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)
        return image

    def preprocess_output(self, outputs):
        '''
        Method to process outputs before feeding them into the next model for
        inference or for the last step of the app.
        '''

        gaze = outputs[0]

        return gaze

