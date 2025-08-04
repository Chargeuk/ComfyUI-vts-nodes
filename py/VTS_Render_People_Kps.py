import math
from typing import List, Optional, NamedTuple, Tuple, Union
import numpy as np
import cv2
import matplotlib
import torch

class Keypoint(NamedTuple):
    x: float
    y: float
    score: float = 1.0
    id: int = -1

class BodyResult(NamedTuple):
    # Note: Using `Optional` instead of `|` operator as the ladder is a Python
    # 3.10 feature.
    # Annotator code should be Python 3.8 Compatible, as controlnet repo uses
    # Python 3.8 environment.
    # https://github.com/lllyasviel/ControlNet/blob/d3284fcd0972c510635a4f5abe2eeb71dc0de524/environment.yaml#L6
    keypoints: List[Optional[Keypoint]]
    total_score: float = 0.0
    total_parts: int = 0

HandResult = List[Keypoint]
FaceResult = List[Keypoint]
AnimalPoseResult = List[Keypoint]
    
class PoseResult(NamedTuple):
    body: BodyResult
    left_hand: Optional[HandResult]
    right_hand: Optional[HandResult]
    face: Optional[FaceResult]

def draw_bodypose(canvas: np.ndarray, keypoints: List[Keypoint], xinsr_stick_scaling: bool = False) -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the body pose.
        keypoints (List[Keypoint]): A list of Keypoint objects representing the body keypoints to be drawn.
        xinsr_stick_scaling (bool): Whether or not scaling stick width for xinsr ControlNet

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn body pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    # force normalised:
    H, W, _ = canvas.shape

    CH, CW, _ = canvas.shape
    stickwidth = 4

    # Ref: https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0
    max_side = max(CW, CH)
    if xinsr_stick_scaling:
        stick_scale = 1 if max_side < 500 else min(2 + (max_side // 1000), 7)
    else:
        stick_scale = 1

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue

        Y = np.array([keypoint1.x, keypoint2.x]) * float(W)
        X = np.array([keypoint1.y, keypoint2.y]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth*stick_scale), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue

        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)

    return canvas

def draw_handpose(canvas: np.ndarray, keypoints: Union[List[Keypoint], None]) -> np.ndarray:
    """
    Draw keypoints and connections representing hand pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the hand pose.
        keypoints (List[Keypoint]| None): A list of Keypoint objects representing the hand keypoints to be drawn
                                          or None if no keypoints are present.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn hand pose.

    """
    if not keypoints:
        return canvas
    
    H, W, _ = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for ie, (e1, e2) in enumerate(edges):
        k1 = keypoints[e1]
        k2 = keypoints[e2]
        if k1 is None or k2 is None:
            continue
        
        x1 = int(k1.x * W)
        y1 = int(k1.y * H)
        x2 = int(k2.x * W)
        y2 = int(k2.y * H)
        # if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
        cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=2)

    for keypoint in keypoints:
        if keypoint is None:
            continue

        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        # if x > eps and y > eps:
        cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas: np.ndarray, keypoints: Union[List[Keypoint], None]) -> np.ndarray:
    """
    Draw keypoints representing face pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the face pose.
        keypoints (List[Keypoint]| None): A list of Keypoint objects representing the face keypoints to be drawn
                                          or None if no keypoints are present.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn face pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """    
    if not keypoints:
        return canvas
    
    H, W, _ = canvas.shape

    for keypoint in keypoints:
        if keypoint is None:
            continue
        
        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas


def draw_poses_separate_canvases(poses: List[PoseResult], H, W, draw_body=True, draw_hand=True, draw_face=True, xinsr_stick_scaling=False) -> List[np.ndarray]:
    """
    Draw the detected poses on separate canvases.

    Args:
        poses (List[PoseResult]): A list of PoseResult objects containing the detected poses.
        H (int): The height of the canvas.
        W (int): The width of the canvas.
        draw_body (bool, optional): Whether to draw body keypoints. Defaults to True.
        draw_hand (bool, optional): Whether to draw hand keypoints. Defaults to True.
        draw_face (bool, optional): Whether to draw face keypoints. Defaults to True.

    Returns:
        List[numpy.ndarray]: A list of 3D numpy arrays representing the canvases with the drawn poses.
    """
    canvases = []

    for pose in poses:
        canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

        if draw_body:
            canvas = draw_bodypose(canvas, pose.body.keypoints, xinsr_stick_scaling)

        if draw_hand:
            canvas = draw_handpose(canvas, pose.left_hand)
            canvas = draw_handpose(canvas, pose.right_hand)

        if draw_face:
            canvas = draw_facepose(canvas, pose.face)

        canvases.append(canvas)

    return canvases

def draw_poses_single_canvas(poses: List[PoseResult], H, W, draw_body=True, draw_hand=True, draw_face=True, xinsr_stick_scaling=False) -> np.ndarray:
    """
    Draw all detected poses on a single canvas.

    Args:
        poses (List[PoseResult]): A list of PoseResult objects containing the detected poses.
        H (int): The height of the canvas.
        W (int): The width of the canvas.
        draw_body (bool, optional): Whether to draw body keypoints. Defaults to True.
        draw_hand (bool, optional): Whether to draw hand keypoints. Defaults to True.
        draw_face (bool, optional): Whether to draw face keypoints. Defaults to True.
        xinsr_stick_scaling (bool, optional): Whether to apply xinsr stick scaling. Defaults to False.

    Returns:
        numpy.ndarray: A 3D numpy array representing the canvas with all poses drawn.
    """
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    for pose in poses:
        if draw_body:
            canvas = draw_bodypose(canvas, pose.body.keypoints, xinsr_stick_scaling)

        if draw_hand:
            canvas = draw_handpose(canvas, pose.left_hand)
            canvas = draw_handpose(canvas, pose.right_hand)

        if draw_face:
            canvas = draw_facepose(canvas, pose.face)

    return canvas

def decode_json_as_poses(
    pose_json: dict,
) -> Tuple[List[PoseResult], List[AnimalPoseResult], int, int]:
    """Decode the json_string complying with the openpose JSON output format
    to poses that controlnet recognizes.
    https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md

    Args:
        json_string: The json string to decode.

    Returns:
        human_poses
        animal_poses
        canvas_height
        canvas_width
    """
    height = pose_json["canvas_height"]
    width = pose_json["canvas_width"]

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def decompress_keypoints(
        numbers: Optional[List[float]],
    ) -> Optional[List[Optional[Keypoint]]]:
        if not numbers:
            return None

        assert len(numbers) % 3 == 0

        def create_keypoint(x, y, c):
            if c < 1.0:
                return None
            keypoint = Keypoint(x, y)
            return keypoint

        return [create_keypoint(x, y, c) for x, y, c in chunks(numbers, n=3)]

    return (
        [
            PoseResult(
                body=BodyResult(
                    keypoints=decompress_keypoints(pose.get("pose_keypoints_2d"))
                ),
                left_hand=decompress_keypoints(pose.get("hand_left_keypoints_2d")),
                right_hand=decompress_keypoints(pose.get("hand_right_keypoints_2d")),
                face=decompress_keypoints(pose.get("face_keypoints_2d")),
            )
            for pose in pose_json.get("people", [])
        ],
        [decompress_keypoints(pose) for pose in pose_json.get("animals", [])],
        height,
        width,
    )



class VTS_Render_People_Kps:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "kps": ("POSE_KEYPOINT",),
                "render_body": ("BOOLEAN", {"default": True}),
                "render_hand": ("BOOLEAN", {"default": True}),
                "render_face": ("BOOLEAN", {"default": True}),
                "max_frames": ("INT", { "default": 0, "min": 0, "step": 1, }),
                "render_combined": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "VTS"
    OUTPUT_IS_LIST = (
        True,
    )

    def render(self, kps, render_body, render_hand, render_face, max_frames, render_combined) -> tuple[np.ndarray]:
        # ensure we are dealing with a list of frames
        if not isinstance(kps, list):
            kps = [kps]

        results = []

        for idx, frame in enumerate(kps):
            poses, _, height, width = decode_json_as_poses(frame)
            
            if render_combined:
                # Render all poses on a single canvas
                combined_canvas = draw_poses_single_canvas(
                    poses,
                    height,
                    width,
                    render_body,
                    render_hand,
                    render_face,
                )
                results.append([combined_canvas])  # Single canvas per frame
            else:
                # Original behavior: separate canvases for each person
                np_images = draw_poses_separate_canvases(
                    poses,
                    height,
                    width,
                    render_body,
                    render_hand,
                    render_face,
                )
                results.append(np_images)
            
        if render_combined:
            # For combined rendering, we have one image per frame
            # Stack all frames into a single tensor
            combined_frames = [frame_images[0] for frame_images in results]
            
            # Truncate to max_frames if specified
            if max_frames > 0:
                combined_frames = combined_frames[:max_frames]
            
            # Convert to tensor
            final_result = [torch.from_numpy(np.stack(combined_frames, axis=0).astype(np.float32) / 255)]
        else:
            # Original character separation logic
            # results is an list of image lists, where the first level represents a frame in time
            # and the second level represents the images for each person in that frame
            # now we need to stack the images for each person seperately
            # Determine the maximum number of characters in any frame
            max_characters = max(len(frame_images) for frame_images in results)

            # Initialize a list to hold the stacked images for each character
            character_stacks = [[] for _ in range(max_characters)]

            # Stack images for each character across all frames
            for frame_images in results:
                for char_idx in range(max_characters):
                    if char_idx < len(frame_images):
                        character_stacks[char_idx].append(frame_images[char_idx])
                    else:
                        # If a character is missing in a frame, append an empty image
                        empty_image = np.zeros_like(frame_images[0])
                        character_stacks[char_idx].append(empty_image)

                # Truncate each character's stack to max_frames if max_frames > 0
            if max_frames > 0:
                character_stacks = [char_stack[:max_frames] for char_stack in character_stacks]

            # Convert the list of character stacks to numpy arrays
            final_result = [torch.from_numpy(np.stack(char_stack, axis=0).astype(np.float32) / 255) for char_stack in character_stacks]

        return (final_result,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Render People Kps": VTS_Render_People_Kps
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Render People Kps": "VTS Render People Kps"
}