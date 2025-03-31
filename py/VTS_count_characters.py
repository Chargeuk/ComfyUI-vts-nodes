from typing import List

class VTS_Count_Characters:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "string_list": ("STRING",),
                            }}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Character_Data",)
    FUNCTION = "upscale"

    INPUT_IS_LIST = True

    OUTPUT_IS_LIST = (
        False,
    )

    CATEGORY = "VTS"

    def upscale(self, string_list: List[str]):
        # as INPUT_IS_LIST is true, all input parameters are lists
        # the only one we want to use as a list is image_list
        # the rest of the parameters are single values

        numberOfWomen = 0
        numberOfMen = 0
        for singleString in string_list:
            lowercaseString = singleString.lower().strip()
            # if the string contains any of the following words:woman, female, lady, set a boolean named isWoman to true
            isWoman = any(word in lowercaseString for word in ["woman", "female", "lady"])
            if isWoman:
                numberOfWomen += 1
            else:
                numberOfMen += 1

        finalResult = ""

        if numberOfWomen > 0:
            womenString = f"There are {numberOfWomen} women in the video."
            finalResult += womenString

        if numberOfMen > 0:
            menString = f"There are {numberOfMen} men in the video."
            if finalResult:  # Add a space if womenString is already included
                finalResult += " "
            finalResult += menString

        return (finalResult,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Count Characters": VTS_Count_Characters
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Count Characters": "VTS Count Characters"
}