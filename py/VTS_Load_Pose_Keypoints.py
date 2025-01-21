import json

class VTS_Load_Pose_Keypoints:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filename": ("STRING", {"placeholder": "X://path/to/images", "vhs_path_extensions": []}),
            },
        }
    RETURN_TYPES = (
        "POSE_KEYPOINT",
    )
    FUNCTION = "load_pose_kps"
    CATEGORY = "VTS"

    def __init__(self):
        self.prefix_append = ""

    def load_pose_kps(self, filename):
        with open(filename, 'r') as f:
            result = json.load(f)
            # as the file contains json data, we can directly load it
            return (result, )


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS_Load_Pose_Keypoints": VTS_Load_Pose_Keypoints
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS_Load_Pose_Keypoints": "VTS Load Pose Keypoints"
}