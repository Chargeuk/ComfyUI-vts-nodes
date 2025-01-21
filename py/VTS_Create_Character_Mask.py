import torch

def conditioning_set_values(conditioning, values={}):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        for k in values:
            n[1][k] = values[k]
        c.append(n)
    return c

class VTS_Create_Character_Mask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                              "type": (["character", "face"],),
                              "number_of_characters": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                             }}

    RETURN_TYPES = (
        "STRING",
    )

    FUNCTION = "create_character_mask"

    CATEGORY = "VTS"


    @staticmethod
    def rgb_to_hex(red: int, green: int, blue: int) -> str:
        # the input colours are in the range of 0-255
        # the output should be a string like #RRGGBB
        return f"#{red:02x}{green:02x}{blue:02x}"

    def create_character_mask(self, type: str, number_of_characters: int):
        red = 255
        green = 0
        blue = 0
        all_character_colors = []
        # count through each character
        for i in range(number_of_characters):
            characterColors = []
            # change the colour
            red = 255 - i

            character_face_colour = VTS_Create_Character_Mask.rgb_to_hex(red, 128, 128)
            characterColors.append(character_face_colour)
            if type == "face":
                character_colours_string = ",".join(characterColors)
                all_character_colors.append(f'"{character_colours_string}"')
                continue

            # convert the colour to hex
            character_body_colour = VTS_Create_Character_Mask.rgb_to_hex(red, 0, 0)
            characterColors.append(character_body_colour)
            
            numberOfClothes = 10
            for j in range(numberOfClothes):
                green = 255 - j
                # convert the colour to hex
                character_clothes_colour = VTS_Create_Character_Mask.rgb_to_hex(red, green, 0)
                characterColors.append(character_clothes_colour)
            numberOfHairs = 10
            for j in range(numberOfHairs):
                blue = 255 - j
                # convert the colour to hex
                character_clothes_colour = VTS_Create_Character_Mask.rgb_to_hex(red, 0, blue)
                characterColors.append(character_clothes_colour)
            character_colours_string = ",".join(characterColors)
            all_character_colors.append(f'"{character_colours_string}"')

        all_character_colors_string = ",".join(all_character_colors)
        result = f"[{all_character_colors_string}]"
        return (result, )

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Create Character Mask": VTS_Create_Character_Mask
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Create Character Mask": "VTS Create Character Mask"
}


