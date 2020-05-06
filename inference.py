#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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

import time
import cv2
import numpy as np
from openvino.inference_engine import IECore
import logging


class Network:
    """
    Load and configure inference plugins for the specified target devices
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self, model, device, batch_size=1):
        """Initialize the parameters"""
        self.model = model
        self.device = device
        self.batch_size = batch_size

    def load_model(self):
        """Load the model"""
        model_weights = self.model + '.bin'
        model_structure = self.model + '.xml'

        t1 = cv2.getTickCount()
        core = IECore()
        self.net = core.read_network(model=model_structure, weights=model_weights)
        self.net.batch_size = self.batch_size
        self.exec_net = core.load_network(network=self.net, device_name=self.device)
        t2 = cv2.getTickCount()
        logging.debug(f'Time taken to load model = {(t2-t1)/cv2.getTickFrequency()} seconds')

        # Get the supported layers of the network
        supported_layers = core.query_network(network=self.net, device_name=self.device)

        # Check for any unsupported layers, and let the user
        # know if anything is missing. Exit the program, if so.
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            logging.error("Unsupported layers found: {}".format(unsupported_layers))
            logging.error("Check whether extensions are available to add to IECore.")
            exit(1)

        # Get the input layer
        self.input_blob = next(iter(self.exec_net.inputs))

        self.output_blob = next(iter(self.exec_net.outputs))
        return

    def get_input_shape(self):
        """Return the input shape"""
        return self.net.inputs[self.input_blob].shape

    def async_exec_net(self, batch):
        """
        Asynchronously run prediction on a batch with the network

        Parameters
        ----------
            batch: the batch of images to perform inference on

        Returns
        -------
            infer_request_handle: the handle for the asynchronous request, needed by async_wait
        """
        infer_request_handle = self.exec_net.start_async(request_id=0, inputs={self.input_blob: batch})
        return infer_request_handle

    def async_wait(self, infer_request_handle):
        """
        Wait for an asynchronous call to finish and return the detections

        Returns
        -------
            detections: the network's detections
        """
        while True:
            status = infer_request_handle.wait(-1)
            if status == 0:
                break
            else:
                time.sleep(1)
        detections = infer_request_handle.outputs
        return detections

    def sync_exec_net(self, batch):
        """
        Synchronously run prediction on a batch with the network

        Parameters
        ----------
            batch: the batch of images to perform inference on

        Returns
        -------
            detections_arr: the array of detections
        """
        t1 = cv2.getTickCount()
        detections = self.exec_net.infer({self.input_blob: batch})
        t2 = cv2.getTickCount()
        logging.debug(f'Time taken to execute model = {(t2-t1)/cv2.getTickFrequency()} seconds')
        return detections

    def get_output(self, detections_arr, threshold=0.3, whitelist_filter=[], normalization_consts=[1.0, 1.0]):
        """
        Change the format of the detections

        Parameters
        ----------
            detections_arr: the tensorflow object detection api network output as produced by OpenVINO, an array of detections
            threshold: discard detections with a score lower than this threshold
            whitelist_filter: the class ids to include, if empty it includes all of them

        Returns
        -------
            detections: a dictionary of detections meeting the criteria
        """
        output = np.concatenate(detections_arr['DetectionOutput'][:, 0, :, :], axis=0)
        # filter based on threshold
        output = output[output[:, 2]>threshold, :]
        #print(output) #TODO bbox output looks wrong for batch size > 1
        #print(output.shape)
        if whitelist_filter:
            output = output[np.isin(output[:, 1], whitelist_filter), :]
        num_detections = output.shape[0]
        return {'num_detections': num_detections,
                'batch': output[:, 0],
                'class': output[:, 1],
                'score': output[:, 2],
                'bbox': output[:, 3:] / np.hstack((normalization_consts, normalization_consts))}

def preprocess_image(image, width: int=640, height: int=640, preserve_aspect_ratio: bool=True):
    """
    Parameters
    ----------
        image: image to run preprocessing on
        width: desired width
        height: desired height
        preserve_aspect_ratio: boolean, https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html specifies for different models

    Returns
    -------
        image: with preprocessing applied
        normalization_consts: ratio of the image pixels to image with padding
    """
    normalization_consts = [1.0, 1.0]
    if preserve_aspect_ratio:
        rows, cols, _ = image.shape
        fx = height * 1.0 / cols
        fy = width * 1.0 / rows
        if fx < fy:
            fy = fx
        else:
            fx = fy
        resized = cv2.resize(image.copy(), (0, 0), fx=fx, fy=fy)
        image = np.zeros((height, width, 3), np.uint8)
        normalization_consts = [resized.shape[1] * 1.0 / image.shape[1],
                                resized.shape[0] * 1.0 / image.shape[0]]
        image[:resized.shape[0], :resized.shape[1], :] = resized
    else:
        image = cv2.resize(image.copy(), (width, height))
    image = image.transpose((2,0,1)) # Channels first
    return image, normalization_consts

def draw_bboxes(image, detections):
    img = image.copy()
    for i in range(detections['batch'].shape[0]):
        #classId = int(detections['class'][i])
        score = float(detections['score'][i])
        bbox = [float(v) for v in detections['bbox'][i]]
        if score > 0.3:
            logging.debug(f"batch: {detections['batch'][i]} class: {classId}, score: {score}, bbox: {bbox}")
            y = bbox[1] * img.shape[0]
            x = bbox[0] * img.shape[1]
            bottom = bbox[3] * img.shape[0]
            right = bbox[2] * img.shape[1]
            cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
    return img

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Use OpenVino to detect objects in an image, uses intermediate representation with tensorflow object detection api')
    parser.add_argument('--filepath', default='resources/image_0100.jpeg', type=str, help='path to the image file')
    parser.add_argument('--model', default='frozen_inference_graph', type=str, help='path to the model excluding the extension')
    parser.add_argument('--device', default='CPU', type=str, help='the device to run inference on, one of CPU, GPU, MYRIAD, FPGA')
    parser.add_argument('--batch-size', default=1, type=int, help='size of the batch')
    parser.add_argument('--preserve-aspect-ratio', default=True, type=bool, help='whether to preserve the aspect ratio')
    parser.add_argument('--log-level',
                        default='error',
                        type=str,
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='the log level, one of debug, info, warning, error, critical')

    args = parser.parse_args()

    LEVELS = {'debug': logging.DEBUG,
              'info': logging.INFO,
              'warning': logging.WARNING,
              'error': logging.ERROR,
              'critical': logging.CRITICAL}
    log_level = LEVELS.get(args.log_level, logging.ERROR)
    logging.basicConfig(level=log_level)

    img = cv2.imread(args.filepath)

    net = Network(args.model, args.device, args.batch_size)
    net.load_model()
    input_shape = net.get_input_shape()
    #print(f'input_shape {input_shape}')

    image, normalization_consts = preprocess_image(img, input_shape[3], input_shape[2], args.preserve_aspect_ratio)
    batch = np.stack((np.squeeze(image), ) * args.batch_size, axis=0)

    # synchronous example
    #detections = net.sync_exec_net(batch)
    #detections = net.sync_exec_net(batch)

    # asynchronous example
    handle = net.async_exec_net(batch)
    detections = net.async_wait(handle)
    detections_dict = net.get_output(detections, normalization_consts=normalization_consts)

    img = draw_bboxes(img, detections_dict)

    cv2.imwrite('ov_od.png', img)
