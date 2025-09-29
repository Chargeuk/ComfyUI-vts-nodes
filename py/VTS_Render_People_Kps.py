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


def draw_poses_separate_canvases(poses: List[PoseResult], H, W, draw_body=True, draw_hand=True, draw_face=True, xinsr_stick_scaling=False, provided_canvases=None) -> List[np.ndarray]:
    """
    Draw the detected poses on separate canvases.

    Args:
        poses (List[PoseResult]): A list of PoseResult objects containing the detected poses.
        H (int): The height of the canvas.
        W (int): The width of the canvas.
        draw_body (bool, optional): Whether to draw body keypoints. Defaults to True.
        draw_hand (bool, optional): Whether to draw hand keypoints. Defaults to True.
        draw_face (bool, optional): Whether to draw face keypoints. Defaults to True.
        xinsr_stick_scaling (bool, optional): Whether to apply xinsr stick scaling. Defaults to False.
        provided_canvases (List[np.ndarray], optional): Pre-existing canvases to draw on. If None, creates new ones.

    Returns:
        List[numpy.ndarray]: A list of 3D numpy arrays representing the canvases with the drawn poses.
    """
    canvases = []

    for i, pose in enumerate(poses):
        # Use provided canvas if available, otherwise create a new one
        if provided_canvases is not None and i < len(provided_canvases):
            canvas = provided_canvases[i].copy()  # Make a copy to avoid modifying the original
        else:
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
    widthOverride: Optional[int] = None,
    heightOverride: Optional[int] = None,
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

    if heightOverride is not None:
        height = heightOverride
    if widthOverride is not None:
        width = widthOverride

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
            },
            "optional": {
                "draw_canvas": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render"
    CATEGORY = "VTS"
    OUTPUT_IS_LIST = (
        True,
    )

    def render(self, kps, render_body, render_hand, render_face, max_frames, render_combined, draw_canvas=None) -> tuple[np.ndarray]:
        # ensure we are dealing with a list of frames
        if not isinstance(kps, list):
            kps = [kps]

        results = []

        # Handle draw_canvas if provided
        canvas_frames = None
        canvas_height = None
        canvas_width = None
        
        if draw_canvas is not None:
            # draw_canvas is expected to be a tensor with shape (B, H, W, C) or (B, H, W)
            canvas_frames = (draw_canvas * 255).byte().numpy()  # Convert from float [0,1] to uint8 [0,255]
            
            # Handle both grayscale (B, H, W) and color (B, H, W, C) inputs
            if len(canvas_frames.shape) == 3:  # Grayscale (B, H, W)
                canvas_height, canvas_width = canvas_frames.shape[1], canvas_frames.shape[2]
                # Expand grayscale to 3 channels by repeating the single channel
                canvas_frames = np.repeat(canvas_frames[:, :, :, np.newaxis], 3, axis=3)
            else:  # Color (B, H, W, C)
                _, canvas_height, canvas_width, channels = canvas_frames.shape
                # If single channel, expand to 3 channels
                if channels == 1:
                    canvas_frames = np.repeat(canvas_frames, 3, axis=3)

        for idx, frame in enumerate(kps):
            # Use canvas dimensions if provided, otherwise use default from pose data
            width_override = canvas_width if canvas_width is not None else None
            height_override = canvas_height if canvas_height is not None else None
            
            poses, _, height, width = decode_json_as_poses(frame, width_override, height_override)
            
            if render_combined:
                # Get canvas for this frame if available
                frame_canvas = None
                if canvas_frames is not None and idx < len(canvas_frames):
                    frame_canvas = canvas_frames[idx]
                
                # Render all poses on a single canvas
                if frame_canvas is not None:
                    # Use provided canvas as base
                    combined_canvas = frame_canvas.copy()
                    for pose in poses:
                        if render_body:
                            combined_canvas = draw_bodypose(combined_canvas, pose.body.keypoints, False)
                        if render_hand:
                            combined_canvas = draw_handpose(combined_canvas, pose.left_hand)
                            combined_canvas = draw_handpose(combined_canvas, pose.right_hand)
                        if render_face:
                            combined_canvas = draw_facepose(combined_canvas, pose.face)
                else:
                    # Use default behavior
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
                frame_canvases = None
                if canvas_frames is not None and idx < len(canvas_frames):
                    # Create separate canvases for each pose from the frame canvas
                    frame_canvases = [canvas_frames[idx] for _ in range(len(poses))]
                
                np_images = draw_poses_separate_canvases(
                    poses,
                    height,
                    width,
                    render_body,
                    render_hand,
                    render_face,
                    provided_canvases=frame_canvases
                )
                results.append(np_images)
            
        if render_combined:
            # For combined rendering, we have one image per frame
            # Stack all frames into a single tensor
            combined_frames = [frame_images[0] for frame_images in results]
            
            # Truncate to max_frames if specified
            if max_frames > 0:
                combined_frames = combined_frames[:max_frames]
            
            # Ensure all frames have the same shape before stacking
            if combined_frames:
                # Get the target shape from the first frame
                target_shape = combined_frames[0].shape
                
                # Check if all frames have the same shape
                shapes_match = all(frame.shape == target_shape for frame in combined_frames)
                
                if not shapes_match:
                    # Find the maximum dimensions across all frames
                    max_height = max(frame.shape[0] for frame in combined_frames)
                    max_width = max(frame.shape[1] for frame in combined_frames)
                    channels = combined_frames[0].shape[2] if len(combined_frames[0].shape) > 2 else 1
                    
                    # Resize all frames to match the maximum dimensions
                    normalized_frames = []
                    for frame in combined_frames:
                        if frame.shape[:2] != (max_height, max_width):
                            # Create a new frame with the target size filled with zeros (black background)
                            if len(frame.shape) == 3:
                                new_frame = np.zeros((max_height, max_width, frame.shape[2]), dtype=frame.dtype)
                                # Copy the original frame into the top-left corner of the new frame
                                h, w = frame.shape[:2]
                                new_frame[:h, :w] = frame
                            else:
                                new_frame = np.zeros((max_height, max_width), dtype=frame.dtype)
                                h, w = frame.shape[:2]
                                new_frame[:h, :w] = frame
                            normalized_frames.append(new_frame)
                        else:
                            normalized_frames.append(frame)
                    combined_frames = normalized_frames
            
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

            # Normalize shapes for each character stack before stacking
            normalized_character_stacks = []
            for char_stack in character_stacks:
                if char_stack:  # Only process non-empty stacks
                    # Check if all images in this character's stack have the same shape
                    target_shape = char_stack[0].shape
                    shapes_match = all(img.shape == target_shape for img in char_stack)
                    
                    if not shapes_match:
                        # Find the maximum dimensions across all images in this character's stack
                        max_height = max(img.shape[0] for img in char_stack)
                        max_width = max(img.shape[1] for img in char_stack)
                        channels = char_stack[0].shape[2] if len(char_stack[0].shape) > 2 else 1
                        
                        # Normalize all images in this character's stack
                        normalized_stack = []
                        for img in char_stack:
                            if img.shape[:2] != (max_height, max_width):
                                # Create a new image with the target size filled with zeros (black background)
                                if len(img.shape) == 3:
                                    new_img = np.zeros((max_height, max_width, img.shape[2]), dtype=img.dtype)
                                    # Copy the original image into the top-left corner of the new image
                                    h, w = img.shape[:2]
                                    new_img[:h, :w] = img
                                else:
                                    new_img = np.zeros((max_height, max_width), dtype=img.dtype)
                                    h, w = img.shape[:2]
                                    new_img[:h, :w] = img
                                normalized_stack.append(new_img)
                            else:
                                normalized_stack.append(img)
                        normalized_character_stacks.append(normalized_stack)
                    else:
                        normalized_character_stacks.append(char_stack)
                else:
                    normalized_character_stacks.append(char_stack)

            # Convert the list of character stacks to numpy arrays
            final_result = [torch.from_numpy(np.stack(char_stack, axis=0).astype(np.float32) / 255) for char_stack in normalized_character_stacks]

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