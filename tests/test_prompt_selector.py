import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("vts_prompt_selector_test", ROOT / "py/VTS_Prompt_Selector.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PromptSelectorTests(unittest.TestCase):
    def setUp(self):
        self.node = MODULE.VTS_Prompt_Selector()

    def test_registration_and_single_string_output(self):
        self.assertIs(MODULE.NODE_CLASS_MAPPINGS["VTS Prompt Selector"], type(self.node))
        self.assertEqual(set(self.node.INPUT_TYPES()["required"]), {"text", "delimiter", "index"})
        self.assertEqual(self.node.RETURN_TYPES, ("STRING",))
        self.assertFalse(getattr(self.node, "OUTPUT_IS_LIST", (False,))[0])
        self.assertEqual(getattr(self.node, self.node.FUNCTION)("cat|dog|bird", "|", 1), ("dog",))

    def test_zero_based_selection_and_last_item_repeat(self):
        for index, expected in [(0, "cat"), (1, "dog"), (2, "bird"), (3, "bird"), (10**12, "bird")]:
            with self.subTest(index=index):
                self.assertEqual(self.node.select_prompt("cat|dog|bird", "|", index), (expected,))

    def test_trim_and_discard_empty_items_before_indexing(self):
        for index, expected in [(0, "cat"), (1, "dog"), (2, "dog")]:
            with self.subTest(index=index):
                self.assertEqual(self.node.select_prompt(" | cat || \n | dog | ", "|", index), (expected,))

    def test_newline_and_multicharacter_delimiters(self):
        for text, delimiter, expected in [("cat\n\ndog\nbird", "\\n", "dog"),
                                           ("cat\r\ndog\r\nbird", "\n", "dog"),
                                           ("cat<->dog<->bird", "<->", "dog"),
                                           ("猫.*犬.*鳥", ".*", "犬")]:
            with self.subTest(delimiter=delimiter):
                self.assertEqual(self.node.select_prompt(text, delimiter, 1), (expected,))

    def test_empty_delimiter_keeps_whole_trimmed_text(self):
        self.assertEqual(self.node.select_prompt(" cat|dog\nbird ", "", 20), ("cat|dog\nbird",))

    def test_empty_input_returns_empty_string(self):
        for text, delimiter in [("", "|"), (" \n ", "|"), ("|||", "|"), ("", ""), ("  ", "")]:
            with self.subTest(text=text, delimiter=delimiter):
                self.assertEqual(self.node.select_prompt(text, delimiter, 100), ("",))

    def test_negative_index_selects_first_item(self):
        self.assertEqual(self.node.select_prompt("cat|dog", "|", -1), ("cat",))


if __name__ == "__main__":
    unittest.main()
