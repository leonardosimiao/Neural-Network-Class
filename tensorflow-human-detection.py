# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import tensorflow as tf
import mss
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

        # print("Elapsed Time:", end_time - start_time)

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

def calculate_intersection(i, classes_i, boxes_i, data):
    #Variable which tells if the hands and objects bounding boxes are being intersected
    # boxes, scores, classes = data
    boxes = [item[0] for item in data]
    scores = [item[1] for item in data]
    classes = [item[2] for item in data]

    ymin_i,xmin_i,ymax_i,xmax_i  = boxes_i[0],boxes_i[1],boxes_i[2],boxes_i[3]
    # intersection = True 

    for ind in range(len(data)):
        # print(len(data))
        
        if ind == i:
            pass
        
        # ymin,xmin,ymax,xmax = boxes[ind][0],boxes[ind][1],boxes[ind][2],boxes[ind][3]
        boxx = boxes[ind]
        ymin = boxx[0]
        xmin = boxx[1] 
        ymax = boxx[2] 
        xmax = boxx[3]
            #Calculation of the intersections
        if ((xmin_i >= xmax) or (xmax_i <= xmin) or (ymin_i >= ymax) or (ymax_i <= ymin)):
            intersection = False
        else:
            intersection = True
            # print(xmin, xmax, ymin, ymax)
            # print(xmin_i, xmax_i, ymin_i, ymax_i)
            # print(classes[ind])

        if classes_i == 2:
            print(f'bike: {xmin_i}, {xmax_i}, {ymin_i}, {ymax_i}')
            print(intersection)
            for item in boxes:
                print(f'{item[1]}, {item[3]}, {item[0]}, {item[2]}')

        return intersection

# def calculate_intersection(i, boxes, classes, num):
#     #Variable which tells if the hands and objects bounding boxes are being intersected

#     ymin_i,xmin_i,ymax_i,xmax_i  = boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
#     intersection = False 

#     for ind in range(num):
        
#         if ind == i:
#             pass
        
#         ymin,xmin,ymax,xmax = boxes[ind][0],boxes[ind][1],boxes[ind][2],boxes[ind][3]

#             #Calculation of the intersections

#             #1 point of the hand's BB is inserted into the object's BB
#         if (xmax_i>xmin and xmax_i<xmax and ymin_i < ymax and ymin_i > ymin):
#             intersection = True
#         if (xmax_i > xmin and xmax_i < xmax and ymax_i > ymin and ymax_i < ymax):
#             intersection = True
#         if (xmin_i < xmax and xmin_i> xmin and ymax_i > ymin and ymax_i < ymax):
#             intersection = True
#         if (xmin_i < xmax and xmin_i > xmin and ymin_i < ymax and ymin_i > ymin):
#             intersection = True

#         #2 points from the object's BB are inserted from the hand's BB
#         if (xmax_i > xmin and xmax_i < xmax and ymax_i > ymax and ymin_i < ymin):
#             intersection = True
#         if (xmin_i < xmin and xmax_i > xmax and ymax_i > ymin and ymax_i < ymax):
#             intersection = True
#             print('sim')
#             if classes[i] == 2:
#                 print('bike sim')
#         if (xmin_i < xmax and xmin_i > xmin and ymin_i < ymin and ymax_i > ymax):
#             intersection = True
#         if (xmin_i < xmin and xmax_i > xmax and ymax_i > ymax and ymin_i < ymax):
#             intersection = True

#         #All points of the object's BB are inserted into the hand's BB
#         if (xmax_i > xmax and xmin_i < xmin and ymax_i > ymax and ymin_i < ymin):
#             intersection = True

#         #All points of the hand's BB are inserted inside the object's BB
#         if (xmax_i < xmax and xmin_i > xmin and ymax_i < ymax and ymin_i > ymin):
#             intersection = True

#         return intersection

if __name__ == "__main__":
    # model_path = 'faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    model_path = '/home/auqua/Neural-Network/human-detection-cnn/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7
    cap = cv2.VideoCapture('/home/auqua/Neural-Network/human-detection-cnn/view-IP1.mp4')
    # bounding_box = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
    prev_frame_time = 0
    new_frame_time = 0

    buffer_human = []
    buffer_object = []
    BUFFER_SIZE = 100

    sct = mss.mss()
    # count = 1
    # while count > 0:
    #     count = -1
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

        data_human = [item for item in zip(boxes[:num], scores[:num], classes[:num]) if (item[2]==1 and item[1]>threshold/2)]
        data_object = [item for item in zip(boxes[:num], scores[:num], classes[:num]) if (item[2]!=1 and item[1]>threshold/2)]

        # print(f'num: {num}, human:{len(data_human)}, obj:{len(data_object)} , class:{sum(classes)}')

        buffer_human = data_human + buffer_human
        if len(buffer_human) >= BUFFER_SIZE:
            buffer_human = buffer_human[:BUFFER_SIZE]
        
        buffer_object = data_object + buffer_object
        if len(buffer_object) >= BUFFER_SIZE:
            buffer_object = buffer_object[:BUFFER_SIZE]
        # for index in range(len(data_full)):
            # print(data_full[index])
        # print(f'Comparacao: {boxes[3]}, {scores[3]}, {classes[3]}')
        # boxes_nonzero = [list_item for list_item in boxes if sum(list_item) != 0]
        # scores_nonzero = [item for item in scores if item != 0]

        # boxes_nonzero = boxes[:num]

        cv2.putText(img, fps, (7, 70), 1, 3, (100, 255, 0), 3, cv2.LINE_AA)

        # Visualization of the results of a detection.

        for i in range(num):  # range(len(boxes)):
            # Changes color of intersecting people and objects
            if classes[i] == 1: # if human
                doesIntersect = calculate_intersection(i, classes[i], boxes[i], buffer_object) # calculate if intersect with object
            else:
                doesIntersect = calculate_intersection(i, classes[i], boxes[i], buffer_human)
            # print(f'ymin={boxes[i][0]},xmin={boxes[i][1]},ymax={boxes[i][2]}, xmax = {boxes[i][3]}')
            # if classes[i] == 2:
                # print(color)
            # Class 1 represents human
            # if classes[i] == 1 and scores[i] > threshold:
            if doesIntersect:
                color = (0, 0, 255)
            else:
                color = (255, 0 , 0)
            
            if classes[i] in TARGET_CLASSES.keys() and scores[i] > threshold:
                box = boxes[i]
                cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), color, 2)
                cv2.putText(img,
                            TARGET_CLASSES.get(classes[i]),
                            (box[1], box[0]),
                            1,
                            2,
                            color,  # (255, 0, 0),
                            2,
                            cv2.LINE_AA)

        cv2.imshow("preview", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
