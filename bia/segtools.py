"""
Authors: Matthew G. French [1]
Affiliations: [1] Auckland Bioengineering Institute

Usage: This module contains generic segmentation tools for the research conducted in
 the Auckland Bioengineering Institute's Breast Biomechanics Research Group.
"""

import numpy as np
import cv2 as cv 


def naively_convert_high_res_contour_to_mask_3d(img_domain, contour_points):
    """
    This function converts the contour based breast visualiser segmentations to a binary mask.

    The contours in the breast visualiser are B-Spline based and can be sampled at any resolution
    by changing numberSplinePointToExport in the configuration file.

    This function uses a very naively method to convert the contours to a binary mask and
    relies on the resolution of the contour (i.e. number of spline points) to be high.

    :param img_domain: shape of image domain (i, j, k) e.g. (100, 100, 100)
    :type img_domain: tuple
    :param contour_points: voxel positions of contour points {point #: {"x": x, "y": y, "z": z}}
    :type contour_points: dict
    :return enclosed_contour_mask: binary mask for segmentation
    :type enclosed_contour_mask: numpy.uint8
    """

    contour_mask = np.zeros(img_domain)

    # find nearest integer co-ordinate for each contour point
    for i in contour_points.keys():
        for j in contour_points[i]:
            contour_mask[round(j["x"]), round(j["y"]), round(j["z"])] = 1

    enclosed_contour_mask = np.zeros(img_domain).astype(np.uint8)

    # iterate through axial plane and enclose contour
    for ax in range(contour_mask.shape[-1]):
        contours, _ = cv.findContours((contour_mask[:, :, ax]*255).astype(np.uint8).copy(), cv.RETR_LIST,
                                      cv.CHAIN_APPROX_SIMPLE)
        enclosed_contour_mask_slice = cv.drawContours((enclosed_contour_mask[:, :, ax]*255).copy(),
                                            [max(contours, key = cv.contourArea)], -1, (255, 255, 255), -1)
        enclosed_contour_mask[:, :, ax] = enclosed_contour_mask_slice

    return enclosed_contour_mask
