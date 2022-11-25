import cv2
import numpy as np


def convert_disparity_to_depth(disparity, reprojection_matrix, min_depth=None, max_depth=None):
    points_3d = cv2.reprojectImageTo3D(disparity, reprojection_matrix)
    depth = points_3d[:, :, -1]

    if min_depth is not None or max_depth is not None:
        depth = np.clip(depth, min_depth, max_depth)
    return depth


def load_stereo_coefficients(path):
    """ Loads stereo matrix coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    K1 = cv_file.getNode("K1").mat()
    D1 = cv_file.getNode("D1").mat()
    K2 = cv_file.getNode("K2").mat()
    D2 = cv_file.getNode("D2").mat()
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    E = cv_file.getNode("E").mat()
    F = cv_file.getNode("F").mat()
    R1 = cv_file.getNode("R1").mat()
    R2 = cv_file.getNode("R2").mat()
    P1 = cv_file.getNode("P1").mat()
    P2 = cv_file.getNode("P2").mat()
    Q = cv_file.getNode("Q").mat()

    cv_file.release()
    return K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q


def draw_depth_map(depth):
    depth_map = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_map = 255 - depth_map
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_TURBO)
    return depth_map

