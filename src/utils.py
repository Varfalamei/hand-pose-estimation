import cv2
import numpy as np
from typing import List, Tuple, Optional, Union

# Define some constants for drawing specs
BLUE_COLOR = (255, 0, 0)
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
YELLOW_COLOR = (0, 255, 255)
CYAN_COLOR = (255, 255, 0)
MAGENTA_COLOR = (255, 0, 255)
WHITE_COLOR = (255, 255, 255)

CONNECTIONS = frozenset({(0, 1),
           (0, 5),
           (0, 17),
           (1, 2),
           (2, 3),
           (3, 4),
           (5, 6),
           (5, 9),
           (6, 7),
           (7, 8),
           (9, 10),
           (9, 13),
           (10, 11),
           (11, 12),
           (13, 14),
           (13, 17),
           (14, 15),
           (15, 16),
           (17, 18),
           (18, 19),
           (19, 20)})


class DrawingSpec:
    def __init__(self, color=WHITE_COLOR, thickness=2, circle_radius=2):
        self.color = color
        self.thickness = thickness
        self.circle_radius = circle_radius


def draw_landmarks(
    image: np.ndarray,
    landmark_list: np.ndarray,
    connections: Union[Tuple[int, int], List[Tuple[int, int]]] = CONNECTIONS,
    landmark_drawing_spec: DrawingSpec = DrawingSpec(color=RED_COLOR),
    connection_drawing_spec: DrawingSpec = DrawingSpec(color=BLUE_COLOR)
):
    # Make sure image is in BGR format
    if image.shape[2] != 3:
        raise ValueError('Input image must contain three channel bgr data.')

    # Draw connections if provided
    if connections is not None:
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            start_point = (int(landmark_list[start_idx][0]), int(landmark_list[start_idx][1]))
            end_point = (int(landmark_list[end_idx][0]), int(landmark_list[end_idx][1]))
            cv2.line(image, start_point, end_point, connection_drawing_spec.color, connection_drawing_spec.thickness)

    # Draw landmarks
    for landmark in landmark_list:
        landmark_point = (int(landmark[0]), int(landmark[1]))
        cv2.circle(image, landmark_point, landmark_drawing_spec.circle_radius, landmark_drawing_spec.color,
                   landmark_drawing_spec.thickness)
