from cv2 import cv2 as cv2
from settings import MAX_MOSAIC
from numpy import ndarray


def assemble_mosaic(figures: list[tuple[ndarray, int]]) -> list[list[ndarray], list[ndarray]]:
    """position cameras in the display according to rank"""
    ranking = [[]]
    max_weight = -1
    max_fig = None
    for figure, weight in figures:
        if weight > max_weight:
            if max_fig is not None:
                ranking[0].append(max_fig)
            max_fig = figure
            max_weight = weight
        else:
            ranking[0].append(figure)
    ranking.append([max_fig])
    return concat_tile_resize(ranking[-MAX_MOSAIC:])


def vconcat_resize_min(
        im_list: list[ndarray],
        interpolation=cv2.INTER_CUBIC
) -> ndarray:
    """Resize all rows of camera images to the same width"""
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)


def hconcat_resize_min(
        im_list: list[ndarray],
        interpolation=cv2.INTER_CUBIC
) -> ndarray:
    """Resize all camera images in a row to the same height"""
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def concat_tile_resize(
        im_list_2d: list[ndarray],
        interpolation=cv2.INTER_CUBIC
) -> ndarray:
    """
    Concatenates camera images into display image.
    
    Each element in im_list_2d represents a row of camera images
    """
    im_list_v = [hconcat_resize_min(im_list_h, interpolation=interpolation) for im_list_h in im_list_2d]
    return vconcat_resize_min(im_list_v, interpolation=interpolation)
