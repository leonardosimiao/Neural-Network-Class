def manage_log_camera(
                cam_index: int, 
                log_overlap: set[str], 
                total_weight: int
) -> str:
    """Creates message that will be published to log, """
    cam_header = f"Camera {cam_index}:\n"
    weight_message = f'Attention Score: {total_weight}\n'
    overlap_header = f'People interacted with: \n'
    footer = '---------------------\n'
    message = cam_header + weight_message + overlap_header
    for class_id in log_overlap:
        message = message + str(f'{class_id}\n')
    message =  message + footer
    return message
