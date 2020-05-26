#!/bin/bash

exec 1>/output/stdout.log 2>/output/stderr.log

F_MODEL=$1
FL_MODEL=$2
HP_MODEL=$3
G_MODEL=$4
VIDEO=$5
DEVICE=$6
OUTPUT=$7

mkdir -p $7

if echo "$DEVICE" | grep -q "FPGA"; then # if device passed in is FPGA, load bitstream to program FPGA
    #Environment variables and compilation for edge compute nodes with FPGAs
    export AOCL_BOARD_PACKAGE_ROOT=/opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/BSP/a10_1150_sg2

    source /opt/altera/aocl-pro-rte/aclrte-linux64/init_opencl.sh
    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/2020-2_PL2_FP16_MobileNet_Clamp.aocx

    export CL_CONTEXT_COMPILER_MODE_INTELFPGA=3
fi

python3 main.py --face_detection_model ${F_MODEL} \
                --facial_landmark_model ${FL_MODEL} \
                --head_pose_model ${HP_MODEL} \
                --gaze_estimation_model ${G_MODEL} \
                --input ${VIDEO} \
                --device ${DEVICE} \
                --output_path ${OUTPUT}

cd /output

tar zcvf output.tgz *