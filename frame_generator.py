import time
from settings import TARGET_CLASSES, threshold
from cv2 import cv2 as cv2
import mss  # only necessary if using screenshots
import numpy as np
from multiprocessing import Queue
from detector import DetectorAPI
from overlap_checker import check_overlap, manage_buffers, highlight_overlap
from log_manager import manage_log_camera

sct = mss.mss()


def get_screenshot(bounding_box: dict) -> None:
    """Takes screenshot of specified region of the screen. Much slower than capturing from video"""
    sct_img = sct.grab(bounding_box)
    img_png = mss.tools.to_png(sct_img.rgb, sct_img.size)
    nparr = np.frombuffer(img_png, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_frame(
        queue: Queue,
        camera: str | dict,
        api: DetectorAPI
) -> None:
    """
    gets frames from pre recorded video or from screenshot from specified bounding box.
    :param queue: Queue object
    :param camera: path to video file or dict specifying region of the screen.
    Example: camera = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080}
    camera = "D:/sample-video.mp4"
    :param api: DetectorAPI object
    :return:
    """
    if isinstance(camera, str):
        cap = cv2.VideoCapture(camera)
    while True:
        if isinstance(camera, str):
            _, img = cap.read()
        elif isinstance(camera, dict):
            img = get_screenshot(camera)
        img = cv2.resize(img, (1280, 720))

        # Process frame
        boxes, scores, classes, num = api.processFrame(img)
        #    boxes, scores, and classes all have size "nmax"
        #    [:num] --> actual detections
        #    [num:nmax] --> garbage
        
        queue.put((img, boxes, scores, classes, num), True)


def generate_frame(
        queue: Queue,
        cam_index: int,
        fig_queue: Queue,
        log: Queue,
        show_fps=True
) -> None:
    """Modify frames from videos to add fps and detections' bounding boxes"""
    
    prev_frame_time = 0
    new_frame_time = 0
    human_buffer = []
    object_buffer = []

    while True:
        if show_fps:
            new_frame_time = time.time()
        
        total_weight = 0
        objects_of_interest = set()  # this set will include all objects whose boxes overlapped people
        img, boxes, scores, classes, num = queue.get(True)

        human_buffer, object_buffer = manage_buffers(boxes, scores, classes, num, human_buffer, object_buffer)
        # Visualization of the results of a detection.

        for i in range(num):
            
            if classes[i] in TARGET_CLASSES.keys() and scores[i] > threshold:
                
                box = boxes[i]
                
                if TARGET_CLASSES.get(classes[i])["id"] == "person":
                    overlap = check_overlap(box_to_check=box, box_buffer=object_buffer)
                else:
                    overlap = check_overlap(box_to_check=box, box_buffer=human_buffer)
                    if overlap:
                        objects_of_interest.add(TARGET_CLASSES.get(classes[i])["id"])
                
                color = highlight_overlap(overlap)

                cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), color, 2)
                cv2.putText(img,
                            TARGET_CLASSES.get(classes[i])["id"],
                            (box[1], box[0]),
                            1,
                            2,
                            color,
                            2,
                            cv2.LINE_AA)
                total_weight += TARGET_CLASSES.get(classes[i])["weight"]

        log_message = manage_log_camera(cam_index, objects_of_interest, total_weight)
        log.put(log_message)

        if show_fps:
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(int(fps))
            cv2.putText(img, fps, (7, 70), 1, 3, (100, 255, 0), 3, cv2.LINE_AA)

        fig_queue.put((img, total_weight), True)
