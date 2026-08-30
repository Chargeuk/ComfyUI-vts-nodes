from pathlib import Path
import json
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


def load_workflow(name):
    with (EXAMPLES / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_graph(testcase, workflow):
    testcase.assertEqual(workflow["version"], 0.4)
    nodes = workflow["nodes"]
    links = workflow["links"]
    node_by_id = {node["id"]: node for node in nodes}
    testcase.assertEqual(len(node_by_id), len(nodes), "node ids must be unique")
    link_by_id = {link[0]: link for link in links}
    testcase.assertEqual(len(link_by_id), len(links), "link ids must be unique")
    testcase.assertGreaterEqual(workflow["last_node_id"], max(node_by_id))
    testcase.assertGreaterEqual(workflow["last_link_id"], max(link_by_id))

    for link_id, origin_id, origin_slot, target_id, target_slot, link_type in links:
        testcase.assertIn(origin_id, node_by_id)
        testcase.assertIn(target_id, node_by_id)
        origin = node_by_id[origin_id]
        target = node_by_id[target_id]
        testcase.assertLess(origin_slot, len(origin["outputs"]))
        testcase.assertLess(target_slot, len(target["inputs"]))
        testcase.assertIn(link_id, origin["outputs"][origin_slot]["links"])
        testcase.assertEqual(target["inputs"][target_slot]["link"], link_id)
        testcase.assertEqual(origin["outputs"][origin_slot]["type"], link_type)
        testcase.assertEqual(target["inputs"][target_slot]["type"], link_type)


class WorkflowTests(unittest.TestCase):
    def test_roundtrip_example_is_closed_and_mask_gated(self):
        workflow = load_workflow("VTS_VR180_Roundtrip_Diagnostic.json")
        validate_graph(self, workflow)
        by_type = {node["type"]: node for node in workflow["nodes"]}
        self.assertIn("VTSRectilinearToVR180", by_type)
        self.assertIn("VTSVR180ToRectilinear", by_type)
        forward = by_type["VTSRectilinearToVR180"]
        reverse = by_type["VTSVR180ToRectilinear"]
        self.assertEqual(forward["outputs"][1]["name"], "known_mask")
        self.assertEqual(reverse["inputs"][1]["name"], "source_known_mask")
        self.assertEqual(reverse["inputs"][1]["link"], 5)
        self.assertEqual(forward["widgets_values"], reverse["widgets_values"])

    def test_ref2va_example_preserves_reference_order_and_inverts_mask_for_reverse(self):
        workflow = load_workflow("VTS_Ref2VA_Projection_Dataset_Eval.json")
        validate_graph(self, workflow)
        nodes = {node["id"]: node for node in workflow["nodes"]}
        ref = next(node for node in workflow["nodes"] if node["type"] == "VTSRef2VAProjectionReferences")
        reverse = next(node for node in workflow["nodes"] if node["type"] == "VTSVR180ToRectilinear")
        inverter = next(node for node in workflow["nodes"] if node["type"] == "InvertMask")
        self.assertEqual([item["name"] for item in ref["outputs"][:3]], ["ref1", "ref2", "effective_mask"])
        self.assertEqual(ref["widgets_values"][0], "16:9")
        self.assertEqual(reverse["widgets_values"][:2], [1024, 576])
        self.assertEqual(inverter["inputs"][0]["link"], 7)
        self.assertEqual(reverse["inputs"][1]["link"], 8)
        self.assertEqual(nodes[2]["outputs"][1]["type"], "IMAGE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
