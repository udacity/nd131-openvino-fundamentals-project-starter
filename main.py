"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import numpy as np
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network, preprocess_image, draw_bboxes

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.3,
                        help="Probability threshold for detections filtering"
                        "(0.3 by default)")
    return parser


def connect_mqtt():
    ### Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network(args.model[:-4], args.device)
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### Load the model through `infer_network` ###
    infer_network.load_model()

    ### Handle the input stream ###
    vc = cv2.VideoCapture(args.input)
    if not vc.isOpened():
        print(f"Error opening input file (video or image {args.input})")
        exit(1)

    ### Read from the video capture ###
    got_frame, frame = vc.read()

    ### Initialize for stats calculation ###
    last_detections = None
    start_frame = None
    total_count = 0

    ### Loop until stream is over ###
    while got_frame:

        ### Pre-process the image as needed ###
        image, normalization_consts = preprocess_image(frame, width=640, height=640, preserve_aspect_ratio=True)
        batch = image[np.newaxis, :, :, :]

        ### Start asynchronous inference for specified request ###
        infer_request_handle = infer_network.async_exec_net(batch)

        ### Wait for the result ###
        detections_arr = infer_network.async_wait(infer_request_handle)

        ### Get the results of the inference request ###
        detections = infer_network.get_output(detections_arr,
                                              threshold=prob_threshold,
                                              whitelist_filter=[1],
                                              normalization_consts=normalization_consts)

        ### Extract any desired stats from the results ###
        ### Calculate and send relevant information on ###
        #TODO improve, use bbox to identify if it is the same person and support multiple people, currently should work for assignment
        current_count = detections['num_detections']
        if current_count > 0 and last_detections:
            if last_detections['num_detections'] > 0:
                duration = (vc.get(cv2.CAP_PROP_POS_MSEC) - start_frame) / 1000.0
            else:
                total_count += current_count
                duration = 0
                start_frame = vc.get(cv2.CAP_PROP_POS_MSEC)
        else:
            duration = 0

        last_detections = detections

        ### current_count, total_count and duration to the MQTT server ###
        ### Topic "person": keys of "count" and "total" ###
        client.publish("person", json.dumps({"count": current_count,
                                             "total": total_count}))
        ### Topic "person/duration": key of "duration" ###
        client.publish("person/duration", json.dumps({"duration": duration}))

        ### Draw bounding boxes to provide intuition ###
        img = draw_bboxes(frame, detections)
        cv2.putText(img,
            f'current: {current_count} total: {total_count} duration: {duration}',
            (0, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            .5,
            (255,255,255),
            2,
            cv2.LINE_AA)

        ## Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(img)
        sys.stdout.flush()

        ### Write an output image if `single_image_mode` ###
        if vc.get(cv2.CAP_PROP_FRAME_COUNT) == 1.0:
            cv2.imwrite('ov_od.png', img)

        ### Read from the video capture ###
        got_frame, frame = vc.read()
    vc.release()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
