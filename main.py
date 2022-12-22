# Code adapted from Tensorflow Object Detection Framework
# https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
# Tensorflow Object Detection Detector

import tensorflow as tf
import cv2.cv2 as cv2
from multiprocessing import Queue, Process
from threading import Thread

from frame_generator import get_frame, generate_frame
from mosaic_assembler import assemble_mosaic
from detector import DetectorAPI
from settings import model_path

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


def display_cameras(queues: list[Queue, ...]) -> None:
    imgs = {}
    while True:
        for index, queue in enumerate(queues):
            imgs.update({index: queue.get(True)})

        if len(imgs.values()) > 1:
            figs = assemble_mosaic(
                [fig for fig in imgs.values()],
            )
        else:
            figs = imgs.get(0)[0]
        img = cv2.resize(figs, (1280, 720))
        cv2.imshow(f"cameras", img)
        key = cv2.waitKey(20)
        if key & 0xFF == ord('q'):
            break


def main(
        camera: str | dict,
        fig_queue: Queue,
) -> None:
    q = Queue(1)
    api = DetectorAPI(model_path)
    threads = [
        Thread(target=get_frame, args=(q, camera, api)),
        Thread(target=generate_frame, args=(q, fig_queue))
    ]
    for t in threads:
        t.start()


if __name__ == "__main__":
    # to terminate processes properly, press "q" on cv2 window
    cameras = [
        'view-IP1.mp4',
        {'top': 0, 'left': 0, 'width': 1920, 'height': 1080},
    ]  # add path to videos or bounding boxes to start processes
    fig_qs = []
    processes = [Process(target=display_cameras, args=(fig_qs,))]
    for cam in cameras:
        fig_qs.append(Queue(1))
        processes.append(Process(target=main, args=(cam, fig_qs[-1])))
    for index, process in enumerate(processes):
        process.start()
        print(f"camera {index} started")
    processes[0].join()  # locks execution until "q" is pressed, then terminates other processes
    for index, process in enumerate(processes[1:], start=1):
        print(f"camera {index} finished")
        # TODO DetectorAPI close function may need to be called before terminating to free up memory
        process.terminate()
