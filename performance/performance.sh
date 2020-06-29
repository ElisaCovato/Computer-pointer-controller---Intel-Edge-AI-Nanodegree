#!/bin/bash

export PYTHONPATH="/opt/intel/openvino/python/python3.6"

DEVICE=$1
PRECISION=$2
INPUT_FILE=$3

if [ "$PRECISION" = "FP32" ]; then
    FACEMODELPATH=../models/intel/face-detection-adas-binary-0001/FP32/face-detection-adas-binary-0001
    POSEMODELPATH=../models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001
    LANDMARKSMODELPATH=../models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009   
    GAZEMODELPATH=../models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-000      
elif [ "$PRECISION" = "FP16" ]; then
    FACEMODELPATH=../models/intel/face-detection-adas-binary-0001/FP16/face-detection-adas-binary-0001
    POSEMODELPATH=../models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001
    LANDMARKSMODELPATH=../models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009   
    GAZEMODELPATH=../models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002
else 
    FACEMODELPATH=../models/intel/face-detection-adas-binary-0001/FP16-INT8/face-detection-adas-binary-0001
    POSEMODELPATH=../models/intel/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001
    LANDMARKSMODELPATH=../models/intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009    
    GAZEMODELPATH=../models/intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002
fi

python3 ../main.py -fm ${FACEMODELPATH} \
                -pm ${POSEMODELPATH} \
                -lm ${LANDMARKSMODELPATH} \
                -gm ${GAZEMODELPATH} \
                -i ${INPUT_FILE} \
                -d ${DEVICE} 
