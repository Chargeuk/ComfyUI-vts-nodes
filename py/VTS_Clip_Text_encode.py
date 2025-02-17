class VTS_Clip_Text_Encode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}), 
                "clip": ("CLIP", {"tooltip": "The CLIP model used for encoding the text."})
            }
        }
    INPUT_IS_LIST = True
    # OUTPUT_IS_LIST = (
    #     True
    # )
    RETURN_TYPES = ("CONDITIONING",)
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "encode"

    CATEGORY = "VTS"
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images."

    @staticmethod
    def encode_text(clip, text):
        tokens = clip.tokenize(text)
        return clip.encode_from_tokens_scheduled(tokens)
    
    def encode(self, clip, text):

        if isinstance(clip, list):
            #print(f"!VTS - clip is a list of length: {len(clip)}")
            clip = clip[0]
        # if text is a list, then process each individually and return a list of conditionings
        if isinstance(text, list):
            #print(f"!VTS - text is a list: {text}")
            return ([VTS_Clip_Text_Encode.encode_text(clip, t) for t in text], )

        # tokenize the text and encode it using the CLIP model
        #print(f"!VTS - text is a string: {text}")
        return (VTS_Clip_Text_Encode.encode_text(clip, text), )

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "VTS Clip Text Encode": VTS_Clip_Text_Encode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS Clip Text Encode": "VTS Clip Text Encode"
}