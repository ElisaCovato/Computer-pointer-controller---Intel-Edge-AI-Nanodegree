import logging as log
import cv2
import sys
from openvino.inference_engine import IECore


class FaceDetectionModel:
    '''
    Class for the Face Detection Model.

    Load and configure inference plugins for the specified target devices,
    and performs either synchronous or asynchronous modes for the
    specified infer requests.
    '''

    def __init__(self, model_name, prob_threshold=0.5, device='CPU', extensions=None, async_infer=True):
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
        self.threshold = prob_threshold
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
            self.network = self.plugin.read_network(model=model_structure, weights=model_weights)
        except:
            raise ValueError("Could not initialise the network. Have you entered the correct model path?")

        # Check if model and CPU plugin are supported
        if self.device == 'CPU':
            self.check_model()

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(network=self.network, device_name=self.device, num_requests=1)

        # Get the input and output layers
        self.input_blob = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_blob].shape
        self.output_blob = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_blob].shape
        return

    def predict(self, image):
        '''
        This method is meant for running predictions on the input image.
        '''
        # Create input image to feed into the network
        net_input = {self.input_blob: self.preprocess_input(image)}

        # Start inference. Infer mode (async/sync) is input by user
        if self.async_infer:
            self.infer_request_handle = self.exec_network.start_async(request_id=0, inputs=net_input)
            # Wait for the result of the inference
            if self.exec_network.requests[0].wait(-1) == 0:
                # Get result of the inference request
                outputs = self.infer_request_handle.outputs[self.output_blob]
        else:
            self.infer_request_handle = self.exec_network.infer(inputs=net_input)
            # Wait for the result of the inference
            if self.exec_network.requests[0].wait(-1) == 0:
                # Get result of the inference request
                outputs = self.infer_request_handle[self.output_blob]

        # Get processed output
        face_coords, crop_face = self.preprocess_output(outputs, image)

        return face_coords, crop_face

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
                log.error(
                    "Please try to specify an extension library path by using the --extensions command line argument.")
            sys.exit(1)
        return

    def preprocess_input(self, image):
        '''
        Method to process inputs before feeding them into the model for inference.
        '''
        image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        image = image.transpose((2, 0, 1))
        image = image.reshape(1, *image.shape)
        return image

    def preprocess_output(self, outputs, image):
        '''
        Method to process outputs before feeding them into the next model for
        inference or for the last step of the app.
        '''
        coords = []
        w = image.shape[1]
        h = image.shape[0]

        for obj in outputs[0][0]:
            # If detection confidence is higher than threshold, return coords
            conf = obj[2]
            if conf > self.threshold:
                coords.append(obj[3:])

        # If no image is detected then set coords and cropped image to empty array:
        if len(coords) == 0:
            face_coords = []
            crop_face = []
        else:
            # Consider only the first detect face to avoid messing up with the
            # mouse controller if more people get into the cam/video stream
            coords = coords[0]
            xmin = int(coords[0] * w)
            ymin = int(coords[1] * h)
            xmax = int(coords[2] * w)
            ymax = int(coords[3] * h)

            face_coords = [xmin, ymin, xmax, ymax]

            # Create cropped image to pass to the next model
            crop_face = image[ymin:ymax, xmin:xmax]

        return face_coords, crop_face