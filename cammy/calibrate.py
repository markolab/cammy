import cv2
import numpy as np

def initialize_board(rows=6, columns=4, marker_size=.043, square_size=.033):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    board = cv2.aruco.CharucoBoard_create(columns, rows, square_size, marker_size, aruco_dict)
    return board


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


def detect_charuco(img, board, refine=True):   
    aruco_corners, aruco_ids, rejected = cv2.aruco.detectMarkers(img, board.dictionary)
    if refine:
        aruco_corners, aruco_ids = cv2.aruco.refineDetectedMarkers(img, board, aruco_corners, aruco_ids, rejected)[:2]

    if len(aruco_corners) > 0:
        ncorners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(aruco_corners, aruco_ids, img, board, minMarkers=0)
    else:
        charuco_corners, charuco_ids = None, None
    return (aruco_corners, aruco_ids), (charuco_corners, charuco_ids)


def estimate_pose(charuco_corners, charuco_ids, intrinsic_matrix, distortion_coeffs, board):
    pose, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners,
        charuco_ids,
        board,
        intrinsic_matrix,
        distortion_coeffs,
        np.empty(1),
        np.empty(1),
        useExtrinsicGuess=False
    )
    return pose, rvec, tvec

