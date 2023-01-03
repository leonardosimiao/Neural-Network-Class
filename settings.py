model_path = 'faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
log_file = 'log_file.txt'
MAX_MOSAIC = 4
BUFFER_SIZE = 20
threshold = 0.7
TARGET_CLASSES = {
    1: {
        "id": "person",
        "weight": 10,
    },
    2: {
        "id": "bicycle",
        "weight": 3,
    },
    27: {
        "id": "backpack",
        "weight": 1,
    },
    28: {
        "id": "umbrella",
        "weight": 2,
    },
    72: {
        "id": "tv",
        "weight": 5,
    },
    73: {
        "id": "laptop",
        "weight": 6
    },
    74: {
        "id": "mouse",
        "weight": 4,
    },
    75: {
        "id": "remote",
        "weight": 1
    },
    76: {
        "id": "keyboard",
        "weight": 1,
    },
    77: {
        "id": "cell phone",
        "weight": 3,
    },
    84: {
        "id": "book",
        "weight": 1,
    },
}
