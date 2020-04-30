# Project Write-Up

## Requirements

The model will be deployed on the edge, such that only data on:
1) the number of people in the frame
2) time those people spent in frame
3) the total number of people counted are sent to a MQTT server


## Stretch Goals

### Suggested stretch goals

* Add an alarm or notification when the app detects above a certain number of people on video, or
people are on camera longer than a certain length of time.
* Try out different models than the People Counter, including a model you have trained. Note that
this may require some alterations to what information is passed through MQTT and what would need to be displayed by the UI.
* Deploy to an IoT device (outside the classroom workspace or your personal computer), such as a
Raspberry Pi with Intel® Neural Compute Stick.
* Add a recognition aspect to your app to be able to tell if a previously counted person returns to
the frame. The recognition model could also be processed on a second piece of hardware.
* Add a toggle to the UI to shut off the camera feed and show stats only (as well as to toggle the
camera feed back on). Show how this affects performance (including network effects) and power.

### Alternative stretch goals

* Add a thermal camera and estimate body temperature of each person, send an alarm for feverish
people, they could be COVID-19 risks (would use the Flir Lepton 2.5 with Purethermal2)
* Add a stereo camera such as the Intel Realsense and estimate distance from the camera
* Compare the throughput with some non-Intel hardware, eg
  * Jetson nano
  * Raspberry Pi cpu 

## Choosing a Model and the Model Optimizer

I wanted to choose a newer more accurate model than the ssd mobilenet v2 that we used in the
course, yet something light enough to run on an edge device such as a cpu.  I found the list of
supported tensorflow models at this link:

https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html

This eliminated some of the more recent developments in tensorflow such as efficient-dets and
spine net.  I chose to go with ssd resnet 50, because an object detection model will make counting
people easy (requirement 1), and I can use the bounding box location to determine if it is the
same person so I can count the time spent in the frame (requirement 2) and differentiate new
people (requirement 3).

The tensorflow object detection model zoo is here:

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

The ssd resnet 50 model is here:

http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz

The commands I used to convert my model to an intermediate representation was:

```bash
wget http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
tar xvf ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz
export MOD_OPT=/opt/intel/openvino/deployment_tools/model_optimizer
export MOD_DIR=ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03

python3 $MOD_OPT/mo.py \
  --input_model $MOD_DIR/frozen_inference_graph.pb \
  --tensorflow_object_detection_api_pipeline_config $MOD_DIR/pipeline.config \
  --reverse_input_channels \
  --transformations_config $MOD_OPT/extensions/front/tf/ssd_v2_support.json
```

## Explaining Custom Layers

The process behind converting custom layers involves...

Some of the potential reasons for handling custom layers are...

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were...

The difference between model accuracy pre- and post-conversion was...

The size of the model pre- and post-conversion was...

The inference time of the model pre- and post-conversion was...

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

Each of these use cases would be useful because...

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
  
- Model 2: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...

- Model 3: [Name]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  - The model was insufficient for the app because...
  - I tried to improve the model for the app by...
