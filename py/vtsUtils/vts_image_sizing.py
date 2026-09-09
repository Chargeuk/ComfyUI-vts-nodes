"""Shared dimension and center-crop rules for VTS image sizing nodes."""


class ScaleToMinDimensions:
    def _center_crop_for_aspect(self, image, width, height):
        old_height, old_width = image.shape[1], image.shape[2]
        old_aspect = old_width / old_height
        new_aspect = width / height

        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)

        if x == 0 and y == 0:
            return image

        cropped = image[:, y:old_height - y, x:old_width - x, :]
        return cropped


    def _calculate_target_dimensions(self, original_width, original_height,
                                     smallMaxSize, largeMaxSize, divisible_by,
                                     scale_type):
        # Treat the values as side limits even if they were entered backwards.
        smallMaxSize, largeMaxSize = sorted((smallMaxSize, largeMaxSize))

        largest_side = max(original_height, original_width)
        smallest_side = min(original_height, original_width)
        aspect_ratio = largest_side / smallest_side

        new_largest_side = round(smallMaxSize * aspect_ratio)
        new_smallest_side = round(largeMaxSize / aspect_ratio)

        if scale_type == "small":
            width, height = self.getSmallDimensions(
                original_width,
                original_height,
                smallMaxSize,
                largeMaxSize,
                new_largest_side,
                new_smallest_side,
            )
            width, height = self._snap_near_aspect_dimensions(
                original_width,
                original_height,
                width,
                height,
                smallMaxSize,
                largeMaxSize,
                divisible_by,
            )
        elif scale_type == "large":
            width, height = self.getLargeDimensions(
                original_width, original_height, smallMaxSize, largeMaxSize)
        else:
            width, height = self.getMaxDimensions(
                original_width, original_height, smallMaxSize, largeMaxSize)

        if divisible_by > 1:
            width -= width % divisible_by
            height -= height % divisible_by

        return width, height

    def _snap_near_aspect_dimensions(self, original_width, original_height,
                                     width, height, smallMaxSize,
                                     largeMaxSize, divisible_by):
        if original_width < original_height:
            target_width, target_height = smallMaxSize, largeMaxSize
        else:
            target_width, target_height = largeMaxSize, smallMaxSize

        tolerance = max(1, divisible_by)
        if (abs(width - target_width) <= tolerance and
                abs(height - target_height) <= tolerance):
            return target_width, target_height

        return width, height

    def getSmallDimensions(self, original_width, original_height, smallMaxSize, largeMaxSize, new_largest_side, new_smallest_side):
        if new_largest_side <= largeMaxSize:
            width = smallMaxSize if original_width < original_height else new_largest_side
            height = smallMaxSize if original_height < original_width else new_largest_side
        else:
            width = new_smallest_side if original_width < original_height else largeMaxSize
            height = new_smallest_side if original_height < original_width else largeMaxSize
        return width, height

    def getLargeDimensions(self, original_width, original_height, smallMaxSize, largeMaxSize):
        if original_width < original_height:
            width = smallMaxSize
            height = largeMaxSize
        else:
            width = largeMaxSize
            height = smallMaxSize
        return width, height
    
    def getMaxDimensions(self, original_width, original_height, smallMaxSize, largeMaxSize):
        if original_width < original_height:
            height = largeMaxSize
            heightRatio = height / original_height
            width = round(original_width * heightRatio)
        else:
            width = largeMaxSize
            widthRatio = width / original_width
            height = round(original_height * widthRatio)
        return width, height

