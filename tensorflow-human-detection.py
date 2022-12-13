# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import numpy as np
import tensorflow as tf
import cv2
import time
import mss
from multiprocessing import Process, Queue
from math import ceil
from time import sleep

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


model_path = '/home/auqua/Neural-Network/human-detection-cnn/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
threshold = 0.7

TARGET_CLASSES = {
    1: "person",
    2: "bicycle",
    3: "car",
    17: "cat",
    18: "dog",
    27: "backpack",
    28: "umbrella",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
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
        self.default_graph.close()


def manager(cam_queue, camera: str = 'view-IP1.mp4', cam_index=0):
    odapi = DetectorAPI(path_to_ckpt=model_path)
    cap = cv2.VideoCapture(camera)
    
    bounding_box = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
    prev_frame_time = 0
    new_frame_time = 0

    buffer_human = []
    buffer_object = []
    BUFFER_SIZE = 20

    # sct = mss.mss()
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

        data_human = [item for item in zip(boxes[:num], scores[:num], classes[:num]) if (item[2]==1 and item[1]>threshold)]
        data_object = [item for item in zip(boxes[:num], scores[:num], classes[:num]) if ((item[2] in TARGET_CLASSES.keys()) and item[2]!=1 and item[1]>threshold)]

        # print(f'num: {num}, human:{len(data_human)}, obj:{len(data_object)} , class:{sum(classes)}')

        buffer_human = data_human + buffer_human
        if len(buffer_human) >= BUFFER_SIZE:
            buffer_human = buffer_human[:BUFFER_SIZE]
        
        buffer_object = data_object + buffer_object
        if len(buffer_object) >= BUFFER_SIZE:
            buffer_object = buffer_object[:BUFFER_SIZE]
        # Visualization of the results of a detection.

        for i in range(num):  # range(len(boxes)):
            if classes[i] == 1: # if human
                doesIntersect = calculate_intersection(i, classes[i], boxes[i], buffer_object) # calculate if intersect with object
            # else:
            elif classes[i] in TARGET_CLASSES.keys():
                doesIntersect = calculate_intersection(i, classes[i], boxes[i], buffer_human)
            
            # Class 1 represents human
            # if classes[i] == 1 and scores[i] > threshold:
            if doesIntersect:
                color = (0, 0, 255)
            else:
                color = (255, 0 , 0)

            # Class 1 represents human
            # if classes[i] == 1 and scores[i] > threshold:
            if classes[i] in TARGET_CLASSES.keys() and scores[i] > threshold:
                box = boxes[i]
                cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), color, 2)
                cv2.putText(img,
                            TARGET_CLASSES.get(classes[i]),
                            (box[1], box[0]),
                            1,
                            2,
                            color,
                            2,
                            cv2.LINE_AA)

        while cam_queue.full():
            pass
            sleep(.1)
        img = cv2.resize(img, [640, 480])
        cam_queue.put(img)


def assemble_layouts(layout, n_figures):
    layouts = []
    for setting in layout:
        if "adjust" in setting:
            diff = 0
            for layout in layouts:
                diff += layout[0] * layout[1]
            remaining_figures = n_figures - diff
            layouts.extend(get_rect_layouts(remaining_figures))
            break
        else:
            layouts.append((int(setting[0]), int(setting[1])))

    diff = n_figures - len(layouts)  # might be dangerous
    if diff and "adjust" not in layout:
        for _ in range(diff):
            layouts.append(layouts[-1])

    return layouts


def get_rect_layouts(n_figures):
    layouts = []
    max_lines = 2
    max_columns = 4
    max_figs = 8
    while n_figures:
        layouts.append(
            (
                max_lines, ceil(n_figures / max_lines) if n_figures < max_figs else max_columns
            )
        )
        n_figures -= n_figures if n_figures < max_figs else max_figs

    return layouts


def get_concat(
        figures,
        layouts
):
    figs = []
    counter = 0
    while figures:
        layout = layouts[counter]
        n_rows = layout[0]
        n_columns = layout[1]
        slice_size = n_rows * n_columns
        fig_slice = figures[:slice_size]
        del figures[:slice_size]
        org = []
        for current_row in range(1, n_rows + 1):
            org.append(fig_slice[n_columns * (current_row - 1):n_columns * current_row])
        figs.append(concat_tile_resize(org))
        counter += 1
    return figs


def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
    im_list_v = [hconcat_resize_min(im_list_h, interpolation=interpolation) for im_list_h in im_list_2d]
    return vconcat_resize_min(im_list_v, interpolation=interpolation)



def calculate_intersection(i, classes_i, boxes_i, data):
    #Variable which tells if the hands and objects bounding boxes are being intersected
    # boxes, scores, classes = data
    boxes = [item[0] for item in data]
    scores = [item[1] for item in data]
    classes = [item[2] for item in data]

    ymin_i,xmin_i,ymax_i,xmax_i  = boxes_i[0],boxes_i[1],boxes_i[2],boxes_i[3]
    

    for ind in range(len(data)):
        
        # ymin,xmin,ymax,xmax = boxes[ind][0],boxes[ind][1],boxes[ind][2],boxes[ind][3]
        boxx = boxes[ind]
        ymin = boxx[0]
        xmin = boxx[1] 
        ymax = boxx[2] 
        xmax = boxx[3]

        #Calculate of the intersections
        if ((xmin_i >= xmax) or (xmax_i <= xmin) or (ymin_i >= ymax) or (ymax_i <= ymin)):
            intersection = False
        else:
            intersection = True
            break

        # if classes_i != 1:
            # print(classes_i)
            # with open('test_boxes.txt', 'w') as file:
            #     file.write('bike')
            #     file.write(f'{xmin_i}, {xmax_i}, {ymin_i}, {ymax_i}')
            #     file.write('\n')
            #     file.write(f'Intersection: {intersection} \n')
            #     for item in boxes:
            #         file.write(f'{item[1]}, {item[3]}, {item[0]}, {item[2]}')
            #         file.write('\n')
            #     file.write('End \n \n \n \n \n')
            # file.close()
    return intersection


if __name__ == "__main__":
    # cameras = ['/home/auqua/Neural-Network/human-detection-cnn/videos/VIRAT_S_000200_01_000226_000268.mp4', '/home/auqua/Neural-Network/human-detection-cnn/videos/VIRAT_S_000004.mp4', '/home/auqua/Neural-Network/human-detection-cnn/videos/VIRAT_S_000102.mp4']  # '/home/auqua/Neural-Network/human-detection-cnn/view-IP1.mp4']
    cameras = ['/home/auqua/Neural-Network/human-detection-cnn/VIRAT_S_010003_00_000000_000108.mp4', '/home/auqua/Neural-Network/human-detection-cnn/videos/VIRAT_S_000200_06_001693_001824.mp4', '/home/auqua/Neural-Network/human-detection-cnn/videos/VIRAT_S_000102.mp4']
    # cameras = ['/home/auqua/Neural-Network/human-detection-cnn/view-IP1.mp4']
    threads = []
    queues = []
    counter = 0
    for cam in cameras:
        queues.append(Queue(1))
        threads.append(Process(target=manager, args=(queues[-1], cam, counter)))
        counter += 1
    for index, thread in enumerate(threads):
        thread.start()
        print(f"camera {index} started")

    videorecorder = cv2.VideoWriter('video_multi.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (1280, 720))
    while True:
        if all([queue.full() for queue in queues]):
            imgs = {}
            for index, queue in enumerate(queues):
                imgs.update({index: queue.get(True)})

            layouts = assemble_layouts(["adjust"], len(imgs.keys()))

            if len(list(imgs.values())) > 1:
                figs = get_concat(
                    [fig for fig in imgs.values()],
                    layouts,
                )
                img = cv2.resize(figs[0], (1280, 720))
            else:
                img = imgs.get(0)
            cv2.imshow(f"cameras", img)
            videorecorder.write(img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
        else:
            sleep(.1)
        
    for thread in threads:
        # print('it is here')
        # thread.join()
        thread.terminate()
        # print('it is here')
    videorecorder.release()
