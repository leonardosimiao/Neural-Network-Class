from settings import BUFFER_SIZE, TARGET_CLASSES, threshold

def check_overlap(
        box_to_check: list[int], 
        box_buffer: list[list[int]]
) -> bool:
    """Checks if a boundary box overlaps any of the boxes in a buffer"""

    # Define limits of the box to check
    ymin,xmin,ymax,xmax  = box_to_check
    # Opencv orders the limits in this way

    overlap = False    

    for box in box_buffer:
        # Define limits of the box in the buffer
        ymin_i, xmin_i, ymax_i, xmax_i = box

        # Check overlap
        if ((xmin >= xmax_i) or (xmax <= xmin_i) or (ymin >= ymax_i) or (ymax <= ymin_i)):
            overlap = False
        else:
            overlap = True
            break

    return overlap


def manage_buffers(
        boxes: list[list[int]],
        score: list[int], 
        classes: list[int], 
        num: int, 
        human_buffer, 
        object_buffer
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Using data from the Detector API, separate boxes into human and object buffers
    :param boxes: list of all boxes detected by API
    :param score: list of box scores
    :param classes: list of box classes
    :param num: number of relevant detections
    :param human_buffer: buffer of detected persons' boundary boxes
    :param object_buffer: buffer of detected objects' boundary boxes
    :return: updated human and object buffers
    """
    
    human_boxes = []
    object_boxes = []

    for i in range(num):
        if score[i] < threshold:  # skip detections that don't pass the threshold
            pass
        if TARGET_CLASSES.get(classes[i])["id"] == "person":  # if it is a person
            human_boxes.append(boxes[i])
        elif classes[i] in TARGET_CLASSES.keys():  # if it is one of the selected object classes
            object_boxes.append(boxes[i]) 

    # Put new detections at the beggining of the buffer.
    # If the buffer exceeds size limit, remove detections at the end
    human_buffer = human_boxes + human_buffer
    if len(human_buffer) >= BUFFER_SIZE:
        human_buffer = human_buffer[:BUFFER_SIZE]
    
    object_buffer = object_boxes + object_buffer
    if len(object_buffer) >= BUFFER_SIZE:
        object_buffer = object_buffer[:BUFFER_SIZE]
    
    return human_buffer, object_buffer

def highlight_overlap(overlap: bool) -> tuple[int, int, int]:
    """Defines bounding box color. Highlights it if there is an overlap"""
    # Opencv color notation: GBR
    if overlap:
        color = (0, 0, 255)  # red
    else:
        color = (255, 0, 0)  # green
    
    return color