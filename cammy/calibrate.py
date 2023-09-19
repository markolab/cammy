import cv2
import numpy as np
from tqdm.auto import tqdm
from typing import Tuple


def initialize_boards(
    squares: Tuple[int, int]=(6, 6),
    marker_length: float=0.043,
    square_length: float=0.033,
    num_slices: int=6,
    markers_per_slice: int=18,
    ar_dict: int=3,
) -> cv2.aruco.CharucoBoard:
    aruco_dict = cv2.aruco.getPredefinedDictionary(ar_dict)
    left_edge = 0
    boards = []
    for slc in tqdm(range(num_slices)):
        aruco_idxs = np.arange(left_edge, left_edge + markers_per_slice)
        boards.append(
            cv2.aruco.CharucoBoard(
                squares, square_length, marker_length, aruco_dict, aruco_idxs
            )
        )
    return boards


def threshold_image(img, invert=True, percentile=95, neighbors=21):
    if img.dtype == "uint16":
        img = (img / 255).astype("uint8")

    if invert:
        threshold = cv2.THRESH_BINARY_INV
    else:
        threshold = cv2.THRESH_BINARY

    img_threshold = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold, percentile, neighbors
    )
    return img_threshold


def detect_charuco(img, boards, refine=True):
    aruco_dat = []
    charuco_dat = []
    for i, _board in enumerate(boards):
        aruco_corners, aruco_ids, rejected = cv2.aruco.detectMarkers(img, _board.dictionary)
        if refine:
            aruco_corners, aruco_ids = cv2.aruco.refineDetectedMarkers(
                img, _board, aruco_corners, aruco_ids, rejected
            )[:2]

        if len(aruco_corners) > 0:
            ncorners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                aruco_corners, aruco_ids, img, boards[use_idx], minMarkers=0
            )
        else:
            charuco_corners, charuco_ids = None, None

        aruco_dat.append((aruco_corners, aruco_ids))
        charuco_dat.append((charuco_corners, charuco_ids))

    return aruco_dat, charuco_dat
    # return boards[use_idx], (aruco_corners, aruco_ids), (charuco_corners, charuco_ids)


def estimate_pose(charuco_corners, charuco_ids, intrinsic_matrix, distortion_coeffs, board):
    pose, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners,
        charuco_ids,
        board,
        intrinsic_matrix,
        distortion_coeffs,
        np.empty(1),
        np.empty(1),
        useExtrinsicGuess=False,
    )
    return pose, rvec, tvec
