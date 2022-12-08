# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import tensorflow as tf
import cv2
import time

TARGET_CLASSES = {  # Classes shown here will be displayed in video
    1: "person",
    2: "bicycle",
    # 3: "car",
    17: "cat",
    18: "dog",
    27: "backpack",
    28: "umbrella",
    # 72: "tv",
    # 73: "laptop",
    # 74: "mouse",
    # 75: "remote",
    # 76: "keyboard",
    # 77: "cell phone",
    84: "book",
}


class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v2.io.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.compat.v1.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time - start_time)

        im_height, im_width, _ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0, i, 0] * im_height),
                             int(boxes[0, i, 1] * im_width),
                             int(boxes[0, i, 2] * im_height),
                             int(boxes[0, i, 3] * im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close

def calculate_intersection(image, camera, detections, xmax_hand, xmin_hand, ymax_hand, ymin_hand, direita = True):
    #Variable which tells if the hands and objects bounding boxes are being intersected
    intersecao = False

    #intersecao_temp has the purpose of comparing if in the previous time step an intersection was detected. isHolding will be true if the intersection disappearead while
    #the object was being intersected with the hand
    global intersecao_temp_right, intersecao_temp_left, isHoldingRight, isHoldingLeft

    for clas, score, box in zip(detections['detection_classes'], detections['detection_scores'], detections['detection_boxes']):
        if (camera.category_index.get(clas).get('name')) == parts[-1] and score > .30:
        # if (camera.category_index.get(clas).get('name')) == pecas[-1] and score > .30:

            #Coordinates of the objects bounding boxes
            ymin,xmin,ymax,xmax = box[0]*camera.resolution[1],box[1]*camera.resolution[0],(box[2])*camera.resolution[1],(box[3])*camera.resolution[0]

            #Calculation of the intersections

            #1 point of the hand's BB is inserted into the object's BB
            if (xmax_hand>xmin and xmax_hand<xmax and ymin_hand < ymax and ymin_hand > ymin):
                intersecao = True
            if (xmax_hand > xmin and xmax_hand < xmax and ymax_hand > ymin and ymax_hand < ymax):
                intersecao = True
            if (xmin_hand < xmax and xmin_hand> xmin and ymax_hand > ymin and ymax_hand < ymax):
                intersecao = True
            if (xmin_hand < xmax and xmin_hand > xmin and ymin_hand < ymax and ymin_hand > ymin):
                intersecao = True

            #2 points from the object's BB are inserted from the hand's BB
            if (xmax_hand > xmin and xmax_hand < xmax and ymax_hand > ymax and ymin_hand < ymin):
                intersecao = True
            if (xmin_hand < xmin and xmax_hand > xmax and ymax_hand > ymin and ymax_hand < ymax):
                intersecao = True
            if (xmin_hand < xmax and xmin_hand > xmin and ymin_hand < ymin and ymax_hand > ymax):
                intersecao = True
            if (xmin_hand < xmin and xmax_hand > xmax and ymax_hand > ymax and ymin_hand < ymax):
                intersecao = True

            #All points of the object's BB are inserted into the hand's BB
            if (xmax_hand > xmax and xmin_hand < xmin and ymax_hand > ymax and ymin_hand < ymin):
                intersecao = True

            #All points of the hand's BB are inserted inside the object's BB
            if (xmax_hand < xmax and xmin_hand > xmin and ymax_hand < ymax and ymin_hand > ymin):
                intersecao = True

            #Drwans on the screen the information of which objects is being hold
            if intersecao == True:
                cv2.rectangle(image, (0, 50), (camera.resolution[0], 80), (50,50,50), -1)
                holding_message = 'Right hand holding ' + str(parts[-1]) if direita else 'Left hand holding ' + str(parts[-1])
                cv2.putText(image, holding_message, (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

if __name__ == "__main__":
    import mss
    # model_path = 'faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    model_path = '/home/auqua/Neural-Network/human-detection-cnn/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    cap = cv2.VideoCapture('/home/auqua/Neural-Network/human-detection-cnn/view-IP1.mp4')
    # bounding_box = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
    prev_frame_time = 0
    new_frame_time = 0

    sct = mss.mss()
    while True:
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))
        r, img = cap.read()
        # sct_img = sct.grab(bounding_box)
        # img_png = mss.tools.to_png(sct_img.rgb, sct_img.size)
        # nparr = np.frombuffer(img_png, np.uint8)
        # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # img = cv2.imread(img_png)
        if img is None:
            break
        img = cv2.resize(img, (1280, 720))
        boxes, scores, classes, num = odapi.processFrame(img)
        cv2.putText(img, fps, (7, 70), 1, 3, (100, 255, 0), 3, cv2.LINE_AA)

        # Visualization of the results of a detection.

        for i in range(len(boxes)):
            # Class 1 represents human
            # if classes[i] == 1 and scores[i] > threshold:
            if classes[i] in TARGET_CLASSES.keys() and scores[i] > threshold:
                box = boxes[i]
                cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
                cv2.putText(img,
                            TARGET_CLASSES.get(classes[i]),
                            (box[1], box[0]),
                            1,
                            2,
                            (255, 0, 0),
                            2,
                            cv2.LINE_AA)

        cv2.imshow("preview", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
