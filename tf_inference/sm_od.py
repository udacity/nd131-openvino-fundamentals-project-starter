import numpy as np
import tensorflow as tf
import cv2
import argparse

parser = argparse.ArgumentParser(description='Use tensorflow framework to detect objects in an image, uses saved model with tensorflow object detection api')
parser.add_argument('--filepath', default='resources/image_0100.jpeg', type=str, help='path to the image file')
parser.add_argument('--model', default='saved_model', type=str, help='path to the model directory')
parser.add_argument('--batch-size', default=10, type=int, help='size of the batch')

args = parser.parse_args()

# Read the graph.
loaded = tf.saved_model.load(export_dir=args.model)
infer = loaded.signatures["serving_default"]
#print(infer.inputs)
#print(infer.outputs)

# Read and preprocess an image.
img = cv2.imread(args.filepath)
rows = img.shape[0]
cols = img.shape[1]

try:
    int(infer.inputs[0].shape[1]) # produces error if not an int, let network resize in that case
    inp = cv2.resize(img, (int(infer.inputs[0].shape[1]), int(infer.inputs[0].shape[1])))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
except:
    inp = img[:, :, [2, 1, 0]]  # BGR2RGB

t1 = cv2.getTickCount()
out = infer(inputs=tf.constant(inp[np.newaxis, :, :, :]))
t2 = cv2.getTickCount()
print('single time ' + str((t2-t1)/cv2.getTickFrequency()))

t1 = cv2.getTickCount()
out = infer(inputs=tf.constant(inp[np.newaxis, :, :, :]))
t2 = cv2.getTickCount()
print('single time (reduced load time) ' + str((t2-t1)/cv2.getTickFrequency()))

batch = np.stack((img,)*args.batch_size)
t1 = cv2.getTickCount()
res = infer(inputs=tf.constant(batch))
t2 = cv2.getTickCount()
print('batch of ' + str(args.batch_size) + ' ' + str((t2-t1)/cv2.getTickFrequency()))

t1 = cv2.getTickCount()
res = infer(inputs=tf.constant(batch))
t2 = cv2.getTickCount()
print('batch of ' + str(args.batch_size) + ' (reduced load time) ' + str((t2-t1)/cv2.getTickFrequency()))

# Visualize detected bounding boxes.
num_detections = int(out['num_detections'].numpy()[0])
for i in range(num_detections):
    classId = int(out['detection_classes'].numpy()[0][i])
    score = float(out['detection_scores'].numpy()[0][i])
    bbox = [float(v) for v in out['detection_boxes'].numpy()[0][i]]
    print(f"class: {classId}, score: {score}, bbox: {bbox}")
    if score > 0.3:
        x = bbox[1] * cols
        y = bbox[0] * rows
        right = bbox[3] * cols
        bottom = bbox[2] * rows
        cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)

cv2.imwrite('sm_od.png', img)

