from comfy_api.latest import io


class VTS_BBox_To_Normalized_Text_List(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VTS_BBox_To_Normalized_Text_List",
            display_name="VTS BBox To Normalized Text List",
            category="VTS/bbox",
            inputs=[
                io.BoundingBox.Input(
                    "bboxes",
                    force_input=True,
                    tooltip="Bounding boxes to normalize. This accepts the 'bboxes' output from nodes like SAM3 Detect.",
                ),
                io.Image.Input(
                    "image",
                    force_input=True,
                    tooltip="Reference image used to determine width and height for normalization.",
                ),
                io.Int.Input(
                    "index",
                    default=-1,
                    min=-1,
                    max=999999,
                    tooltip="Bounding box index to convert after flattening all boxes in order. Use -1 to convert every bbox.",
                ),
                io.Int.Input(
                    "decimal_places",
                    default=6,
                    min=0,
                    max=12,
                    tooltip="Number of decimal places to keep in the normalized output text.",
                ),
            ],
            outputs=[
                io.String.Output("bbox_text", is_output_list=True),
                io.Int.Output("bbox_count"),
            ],
        )

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

    @classmethod
    def execute(cls, bboxes, image, index, decimal_places):
        width, height = cls._get_image_size(image)
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image dimensions for bbox normalization: width={width}, height={height}")

        frame_groups = cls._normalize_bbox_input(bboxes)
        flat_boxes = []
        for frame_boxes in frame_groups:
            flat_boxes.extend(frame_boxes)

        if index >= 0:
            if index >= len(flat_boxes):
                print(
                    f"[VTS BBox To Normalized Text List] Requested bbox index {index} "
                    f"but only {len(flat_boxes)} bounding boxes were available"
                )
                return io.NodeOutput([], 0)
            boxes_to_convert = [flat_boxes[index]]
        else:
            boxes_to_convert = flat_boxes

        output = []

        for box in boxes_to_convert:
            x1, y1, x2, y2 = cls._bbox_to_xyxy(box)
            nx1 = round(cls._clamp01(x1 / width), decimal_places)
            ny1 = round(cls._clamp01(y1 / height), decimal_places)
            nx2 = round(cls._clamp01(x2 / width), decimal_places)
            ny2 = round(cls._clamp01(y2 / height), decimal_places)
            output.append(f"{nx1},{ny1},{nx2},{ny2}")

        print(
            f"[VTS BBox To Normalized Text List] Converted {len(output)} bounding boxes "
            f"using image size {int(width)}x{int(height)} with index={index}"
        )
        return io.NodeOutput(output, len(output))


NODE_CLASS_MAPPINGS = {
    "VTS_BBox_To_Normalized_Text_List": VTS_BBox_To_Normalized_Text_List,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS_BBox_To_Normalized_Text_List": "VTS BBox To Normalized Text List",
}
