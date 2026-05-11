class VTS_BBox_To_Normalized_Text_List:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image": ("IMAGE",),
                "decimal_places": ("INT", {"default": 6, "min": 0, "max": 12}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("bbox_text", "bbox_count")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "convert"
    CATEGORY = "VTS/bbox"

    @staticmethod
    def _get_image_size(image):
        shape = getattr(image, "shape", None)
        if shape is None or len(shape) < 3:
            raise ValueError("Image shape metadata is not available for bbox normalization")
        return float(shape[2]), float(shape[1])  # width, height

    @staticmethod
    def _normalize_bbox_input(bboxes):
        if bboxes is None:
            return []
        if isinstance(bboxes, dict):
            return [[bboxes]]
        if not isinstance(bboxes, list):
            return []
        if len(bboxes) == 0:
            return []
        first = bboxes[0]
        if isinstance(first, list):
            return bboxes
        return [bboxes]

    @staticmethod
    def _bbox_to_xyxy(box):
        if isinstance(box, dict):
            if all(key in box for key in ("x", "y", "width", "height")):
                x1 = float(box["x"])
                y1 = float(box["y"])
                x2 = x1 + float(box["width"])
                y2 = y1 + float(box["height"])
                return x1, y1, x2, y2
            if all(key in box for key in ("x1", "y1", "x2", "y2")):
                return float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
            raise ValueError(f"Unsupported bbox dict keys: {sorted(box.keys())}")

        if isinstance(box, (list, tuple)) and len(box) >= 4:
            x1 = float(box[0])
            y1 = float(box[1])
            x2 = x1 + float(box[2])
            y2 = y1 + float(box[3])
            return x1, y1, x2, y2

        raise ValueError(f"Unsupported bbox value: {box}")

    @staticmethod
    def _clamp01(value):
        return min(1.0, max(0.0, value))

    def convert(self, bboxes, image, decimal_places):
        width, height = self._get_image_size(image)
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions for bbox normalization: width={width}, height={height}")

        frame_groups = self._normalize_bbox_input(bboxes)
        output = []

        for frame_boxes in frame_groups:
            for box in frame_boxes:
                x1, y1, x2, y2 = self._bbox_to_xyxy(box)
                nx1 = round(self._clamp01(x1 / width), decimal_places)
                ny1 = round(self._clamp01(y1 / height), decimal_places)
                nx2 = round(self._clamp01(x2 / width), decimal_places)
                ny2 = round(self._clamp01(y2 / height), decimal_places)
                output.append(f"{nx1},{ny1},{nx2},{ny2}")

        print(f"[VTS BBox To Normalized Text List] Converted {len(output)} bounding boxes using image size {int(width)}x{int(height)}")
        return (output, len(output))


NODE_CLASS_MAPPINGS = {
    "VTS BBox To Normalized Text List": VTS_BBox_To_Normalized_Text_List,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS BBox To Normalized Text List": "VTS BBox To Normalized Text List",
}
