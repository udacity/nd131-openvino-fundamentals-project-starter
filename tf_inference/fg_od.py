import numpy as np
import tensorflow as tf
import cv2
import argparse

parser = argparse.ArgumentParser(description='Use tensorflow framework to detect objects in an image, uses frozen graph with tensorflow object detection api')
parser.add_argument('--filepath', default='resources/image_0100.jpeg', type=str, help='path to the image file')
parser.add_argument('--model', default='frozen_inference_graph.pb', type=str, help='path to the model')
parser.add_argument('--batch-size', default=10, type=int, help='size of the batch')

args = parser.parse_args()

# Read the graph.
with tf.gfile.FastGFile(args.model, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # Read and preprocess an image.
    img = cv2.imread(args.filepath)
    rows = img.shape[0]
    cols = img.shape[1]

    input_tensor = sess.graph.get_tensor_by_name('image_tensor:0')
    try:
        int(input_tensor.shape[1]) # produces error if not an int, let network resize in that case
        inp = cv2.resize(img, (int(input_tensor.shape[1]), int(input_tensor.shape[2])))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
    except:
        inp = img[:, :, [2, 1, 0]]  # BGR2RGB

    # Run the model
    t1 = cv2.getTickCount()
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp[np.newaxis, :, :, :]})
    t2 = cv2.getTickCount()
    print('single time ' + str((t2-t1)/cv2.getTickFrequency()))

    t1 = cv2.getTickCount()
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp[np.newaxis, :, :, :]})
    t2 = cv2.getTickCount()
    print('single time (reduced load time) ' + str((t2-t1)/cv2.getTickFrequency()))

    batch = np.stack((img,)*args.batch_size)
    t1 = cv2.getTickCount()
    res = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': batch})
    t2 = cv2.getTickCount()
    print('batch of ' + str(args.batch_size) + ' ' + str((t2-t1)/cv2.getTickFrequency()))

    t1 = cv2.getTickCount()
    res = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': batch})
    t2 = cv2.getTickCount()
    print('batch of ' + str(args.batch_size) + ' (reduced load time) ' + str((t2-t1)/cv2.getTickFrequency()))

    # Visualize detected bounding boxes.
    num_detections = int(out[0][0])
    for i in range(num_detections):
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        #print(f"class: {classId}, score: {score}, bbox: {bbox}")
        if score > 0.3:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            cv2.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)

cv2.imwrite('fg_od.png', img)

