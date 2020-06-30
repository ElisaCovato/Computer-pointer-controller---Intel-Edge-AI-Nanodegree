# Computer Pointer Controller
Project 3 of the **Intel® Edge AI for IoT Developers** Nanodegree Program.

### The project
The aim of the project is to develop an application that allows users to control the mouse pointer of their computers using their **eyes gaze**, captured through a webcam or a video file.

Using the _InferenceEngine API_ from Intel's OpenVino ToolKit, the application takes the output of a [gaze estimation model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html) to move the mouse pointer. The gaze estimation model requires three inputs:

- The head pose
- The left eye image
- The right eye image.

To get these inputs, three other OpenVino models are used within the application:

- [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_0001_description_face_detection_adas_0001.html)
- [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
- [Facial Landmarks Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)

The flow of data from the input, then amongst the different models and finally to the mouse controller looks like this:

![mouse_controller_data_flow](./media/data_flow.png)

The mouse pointer is controlled using the Python library [PyAutoGui](https://pypi.org/project/PyAutoGUI/).

## Requirements

### Hardware

* 6th to 10th generation Intel® Core™ processor with Iris® Pro graphics or Intel® HD Graphics.
* OR use of Intel® Neural Compute Stick 2 (NCS2)

### Software

*   Intel® Distribution of OpenVINO™ toolkit latest release
*   CMake
*   Python 3.5 to 3.7

## Project Set Up and Installation
To run the application, stick to the following steps:
#### Install Intel® Distribution of OpenVINO™ toolkit

Refer to the relevant instructions for the appropriate operating system [here](https://docs.openvinotoolkit.org/latest/index.html).

#### Clone the directory and install dependencies
Clone this directory:
```
git clone https://github.com/ElisaCovato/Computer-pointer-controller---Intel-Edge-AI-Nanodegree.git
```
The project directory contains a `media` folder which has a demo.mp4 file, that  can be used as input for the project. The `src` folder contains all the python scripts necessary to make `main.py` work. The `performance` folder contains some analysis on the application performance using different devices and/or different model precisions.

After cloning the directory, some python modules need to be installed. To do so, run :
```
pip3 install -r requirements.txt
```

#### Initialize OpenVino environments
Configure the build environment for the OpenVino toolkit by sourcing the `setupvars.sh` script:
```
source /opt/intel/openvino/bin/setupvars.sh
```
If successful, the terminal will prompt  `[setupvars.sh] OpenVINO environment initialized`.

#### Download models
To download the models, run the `dowload_model.sh` bash script:
```
bash download_model.sh
```
The script will create a folder `models` in the main directory containing the four models linked above. Each model is downloaded with its 3 different precision weights: FP32, FP16 and INT8. 

Note that the script makes use of the [Model Downloader](https://docs.openvinotoolkit.org/latest/_tools_downloader_README.html) included in the OpenVino toolkit. Alternatively, it is possible to download the models directly from the [Open Model Zoo Directory](https://download.01.org/openvinotoolkit/2018_R5/open_model_zoo/). It will then suffice to move the .xml and .bin files from the main application directory into `./model/intel/<model_name>/<model_precision>/`, e.g., `./models/intel/face-detection-adas-0001/FP16/`


## Demo
*TODO:* Explain how to run a basic demo of your model.

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.

## License
[MIT License](LICENSE.MIT)