#!/bin/bash

dir_model="./models/"

script_download="/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py"

face_detection="face-detection-adas-0001"
head_position="head-pose-estimation-adas-0001"
facial_landmark="facial-landmarks-35-adas-0002"
gaze_estimation="gaze-estimation-adas-0002"


#Create a model directory
echo "Creating model directory"
mkdir -p ${dir_model}

# Download face-detection-adas-binary-0001
echo "Downloading Face Detection Model"
python3 ${script_download} --name ${face_detection} -o ${dir_model}

# Download head-pose-estimation-adas-0001
echo "Downloading Head Pose Estimation Model"
python3 ${script_download} --name ${head_position} -o ${dir_model}

# Download facial-landmarks-35-adas-0002
echo "Downloading Facial Landmarks Model"
python3 ${script_download} --name ${facial_landmark} -o ${dir_model}

# Download gaze-estimation-adas-0002
echo "Downloading Gaze Estimation model"
python3 ${script_download} --name ${gaze_estimation} -o ${dir_model}

echo "Download Completed"