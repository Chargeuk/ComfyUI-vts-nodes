import random


class VTS_SeedList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_num": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "max_num": ("INT", {"default": 0xFFFFFFFFFFFFFFFF, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "method": (["random", "increment", "decrement"], {"default": "random"}),
                "total": ("INT", {"default": 1, "min": 1, "max": 100000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("seed", "total")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "build_seed_list"
    CATEGORY = "VTS/seed"

    def build_seed_list(self, min_num, max_num, method, total, seed):
        if min_num > max_num:
            min_num, max_num = max_num, min_num

        rng = random.Random(seed)
        seeds = []
        for index in range(total):
            if method == "random":
                value = rng.randint(min_num, max_num)
            elif method == "increment":
                value = min_num + index
                if value > max_num:
                    value = max_num
            else:
                value = max_num - index
                if value < min_num:
                    value = min_num
            seeds.append(value)

        print(
            f"[VTS SeedList] Generated {len(seeds)} seeds using method='{method}' "
            f"range=({min_num}, {max_num}) base_seed={seed}"
        )
        return (seeds, total)


NODE_CLASS_MAPPINGS = {
    "VTS SeedList": VTS_SeedList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VTS SeedList": "VTS SeedList",
}
