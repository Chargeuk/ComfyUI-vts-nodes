"""CPU-only integration tests against the installed ComfyUI scheduler."""
import gc
import asyncio
import importlib.util
import os
from pathlib import Path
import sys
import unittest
import uuid
import weakref
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("COMFYUI_ROOT", str(Path(__file__).resolve().parents[3])))
import comfy.cli_args
comfy.cli_args.args.cpu = True
import torch
import nodes
import execution
from comfy_execution.progress import get_progress_state

spec = importlib.util.spec_from_file_location("vts_memory_loop_test", Path(__file__).parents[1] / "py" / "VTS_MemoryAwareLoop.py")
loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loop)
for node in loop.NODE_CLASS_MAPPINGS.values():
    node.GET_SCHEMA()
nodes.NODE_CLASS_MAPPINGS.update(loop.NODE_CLASS_MAPPINGS)

EVENTS = []
REFS = []
RESULTS = []
ALIVE = []
STORE_COUNTS = []


class Source:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("INT",)}}
    RETURN_TYPES = ("*", "*")
    FUNCTION = "execute"
    def execute(self, value):
        EVENTS.append(("source", value))
        result = torch.tensor([float(value)])
        REFS.append(weakref.ref(result))
        return result, value


class Body:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("*",), "offset": ("*",)}}
    RETURN_TYPES = ("*",)
    FUNCTION = "execute"
    def execute(self, value, offset):
        gc.collect()
        ALIVE.append(sum(ref() is not None for ref in REFS))
        STORE_COUNTS.append(len(loop._LIFECYCLE.current().values))
        result = value + offset
        EVENTS.append(("body", result.item()))
        REFS.append(weakref.ref(result))
        return (result,)


class Sink:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("*",)}}
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "execute"
    def execute(self, value):
        RESULTS.append(value.item() if torch.is_tensor(value) else value)
        return ()


class Barrier(Sink):
    RETURN_TYPES = ("*",)
    def execute(self, value):
        EVENTS.append(("barrier", value.item()))
        return (True,)


class Failure(Body):
    def execute(self, value, offset):
        raise RuntimeError("intentional test failure")


class Cancelled(Body):
    def execute(self, value, offset):
        raise comfy.model_management.InterruptProcessingException()


class LazyBody(Body):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": ("*",), "offset": ("*", {"lazy": True})}}
    def check_lazy_status(self, **kwargs):
        return []


class Server:
    client_id = None
    last_node_id = None
    def send_sync(self, *args, **kwargs):
        pass


nodes.NODE_CLASS_MAPPINGS.update({"TestLoopSource": Source, "TestLoopBody": Body,
    "TestLoopSink": Sink, "TestLoopBarrier": Barrier, "TestLoopFailure": Failure,
    "TestLoopCancelled": Cancelled, "TestLoopLazyBody": LazyBody})


def node(kind, **inputs):
    if kind == loop.END:
        inputs = dict(release_previous=True, collect_garbage=False,
                      empty_cuda_cache=False, trim_cpu=False, cleanup_every=1,
                      log_memory=False, **inputs)
    return {"class_type": kind, "inputs": inputs}


def prompt(total=4):
    return {
        "source": node("TestLoopSource", value=1),
        "start": node(loop.START, total=total, **{"initial.item0": ["source", 0]}),
        "read": node(loop.VALUE, values=["start", 2], key="item0"),
        "body": node("TestLoopBody", value=["read", 0], offset=["source", 1]),
        "end": node(loop.END, flow=["start", 0], **{"values.item0": ["body", 0]}),
        "final": node(loop.VALUE, values=["end", 0], key="item0"),
        "sink": node("TestLoopSink", value=["final", 0]),
    }


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        EVENTS.clear()
        REFS.clear()
        RESULTS.clear()
        ALIVE.clear()
        STORE_COUNTS.clear()

    def run_prompt(self, graph, mode, executor=None):
        executor = executor or execution.PromptExecutor(Server(), cache_type=mode,
            cache_args={"ram": 0, "ram_inactive": 0, "lru": 0})
        outputs = [key for key, value in graph.items()
                   if getattr(nodes.NODE_CLASS_MAPPINGS[value["class_type"]], "OUTPUT_NODE", False)]
        executor.execute(graph, str(uuid.uuid4()), execute_outputs=outputs)
        errors = [message for kind, message in executor.status_messages if kind == "execution_error"]
        self.assertFalse(errors)
        self.assertTrue(executor.success, errors)
        self.assertFalse(loop._LIFECYCLE.runs)
        self.assertIsNone(loop._LIFECYCLE.context.get())
        self.assertFalse(torch.cuda.is_initialized())
        return executor

    def test_four_iterations_both_cache_modes(self):
        for mode in (execution.CacheType.NONE, execution.CacheType.RAM_PRESSURE):
            with self.subTest(mode=mode):
                EVENTS.clear()
                RESULTS.clear()
                self.run_prompt(prompt(), mode)
                self.assertEqual(RESULTS, [5])
                self.assertEqual([x for x in EVENTS if x[0] == "source"], [("source", 1)])
                self.assertEqual([x[1] for x in EVENTS if x[0] == "body"], [2, 3, 4, 5])

    def test_registration_and_validation(self):
        self.assertEqual(tuple(loop.VTSMemoryAwareLoopStart.RETURN_TYPES), ("VTS_LOOP_FLOW", "INT", "VTS_LOOP_VALUES"))
        for cls, prefix in ((loop.VTSMemoryAwareLoopStart, "initial"), (loop.VTSMemoryAwareLoopEnd, "values")):
            slots = cls.INPUT_TYPES()["optional"]
            self.assertTrue(slots[prefix + ".item0"][1]["lazy"])
            self.assertTrue(slots[prefix + ".item99"][1]["rawLink"])
            self.assertNotIn(prefix + ".item100", slots)
            self.assertNotIn(prefix + ".itemfoo", slots)
        result = asyncio.run(execution.validate_prompt(str(uuid.uuid4()), prompt(), None))
        self.assertTrue(result[0], result)
        self.assertFalse(result[3], result)

    def test_shared_original_output_and_barrier(self):
        for mode in (execution.CacheType.NONE, execution.CacheType.RAM_PRESSURE):
            with self.subTest(mode=mode):
                EVENTS.clear()
                graph = prompt()
                graph["original_output"] = node("TestLoopBarrier", value=["source", 0])
                graph["barrier"] = node("TestLoopBarrier", value=["body", 0])
                graph["end"]["inputs"]["after.item0"] = ["barrier", 0]
                self.run_prompt(graph, mode)
                self.assertEqual(sum(x[0] == "source" for x in EVENTS), 1)
                self.assertEqual([x[1] for x in EVENTS if x[0] == "barrier" and x[1] != 1], [2, 3, 4, 5])

    def test_nested(self):
        for mode in (execution.CacheType.NONE, execution.CacheType.RAM_PRESSURE):
            with self.subTest(mode=mode):
                EVENTS.clear()
                RESULTS.clear()
                STORE_COUNTS.clear()
                graph = prompt(3)
                graph["inner_start"] = node(loop.START, total=3, **{"initial.item0": ["read", 0]})
                # Presentation identities deliberately collide; execution ownership must not.
                graph["start"]["override_display_id"] = "same_display"
                graph["inner_start"]["override_display_id"] = "same_display"
                graph["inner_read"] = node(loop.VALUE, values=["inner_start", 2], key="item0")
                graph["body"]["inputs"]["value"] = ["inner_read", 0]
                graph["inner_end"] = node(loop.END, flow=["inner_start", 0], **{"values.item0": ["body", 0]})
                graph["inner_final"] = node(loop.VALUE, values=["inner_end", 0], key="item0")
                graph["end"]["inputs"]["values.item0"] = ["inner_final", 0]
                self.run_prompt(graph, mode)
                self.assertEqual(RESULTS, [10])
                self.assertEqual(sum(x[0] == "source" for x in EVENTS), 1)
                self.assertEqual(sum(x[0] == "body" for x in EVENTS), 9)
                self.assertLessEqual(max(STORE_COUNTS), 2)

    def test_cache_none_tensor_release_and_graph_literals(self):
        self.run_prompt(prompt(12), execution.CacheType.NONE)
        gc.collect()
        self.assertFalse(any(ref() is not None for ref in REFS))
        self.assertLessEqual(max(ALIVE[2:]), 2)
        def check(value):
            self.assertFalse(torch.is_tensor(value))
            if isinstance(value, dict):
                for child in value.values():
                    check(child)
            elif isinstance(value, (tuple, list)):
                for child in value:
                    check(child)
        check(get_progress_state().dynprompt.ephemeral_prompt)

    def test_repeat_prompt_fresh_execution(self):
        graph = prompt(2)
        executor = self.run_prompt(graph, execution.CacheType.RAM_PRESSURE)
        self.run_prompt(prompt(2), execution.CacheType.RAM_PRESSURE, executor)
        self.assertEqual(RESULTS, [3, 3])

    def test_no_carry_index_only(self):
        graph = {"start": node(loop.START, total=4),
                 "end": node(loop.END, flow=["start", 0])}
        self.run_prompt(graph, execution.CacheType.NONE)

    def test_one_iteration_and_release_disabled(self):
        graph = prompt(1)
        graph["end"]["inputs"]["release_previous"] = False
        self.run_prompt(graph, execution.CacheType.NONE)
        self.assertEqual(RESULTS, [2])

    def test_cleanup_cadence_does_not_initialize_cuda(self):
        with patch.object(loop.gc, "collect") as collect, patch.object(loop.model_management, "soft_empty_cache") as empty:
            loop._cleanup(0, 2, True, True, False, False)
            collect.assert_not_called()
            loop._cleanup(1, 2, True, True, False, False)
            collect.assert_called_once()
            empty.assert_not_called()
        self.assertFalse(torch.cuda.is_initialized())
        with self.assertRaisesRegex(ValueError, "at least one"):
            loop._cleanup(0, 0, False, False, False, False)

    def test_terminal_invariants_released_before_cleanup(self):
        for release_previous in (True, False):
            with self.subTest(release_previous=release_previous):
                observed = []
                def observe_cleanup(iteration, *args):
                    observed.append((iteration, len(loop._LIFECYCLE.current().invariants)))
                graph = prompt(2)
                graph["end"]["inputs"]["release_previous"] = release_previous
                with patch.object(loop, "_cleanup", side_effect=observe_cleanup):
                    self.run_prompt(graph, execution.CacheType.NONE)
                self.assertEqual(observed, [(0, 1), (1, 0 if release_previous else 1)])

    def test_failure_clears_execution_store(self):
        for kind in ("TestLoopFailure", "TestLoopCancelled"):
            with self.subTest(kind=kind):
                graph = prompt()
                graph["body"]["class_type"] = kind
                executor = execution.PromptExecutor(Server(), cache_type=execution.CacheType.NONE,
                    cache_args={"ram": 0, "ram_inactive": 0, "lru": 0})
                with self.assertLogs(level="INFO"):
                    executor.execute(graph, str(uuid.uuid4()), execute_outputs=["start", "end", "sink"])
                self.assertFalse(executor.success)
                self.assertFalse(loop._LIFECYCLE.runs)
                self.assertIsNone(loop._LIFECYCLE.context.get())

    def test_lazy_body_rejected_before_unused_source(self):
        graph = prompt()
        graph["unused"] = node("TestLoopSource", value=99)
        graph["body"]["class_type"] = "TestLoopLazyBody"
        graph["body"]["inputs"]["offset"] = ["unused", 1]
        executor = execution.PromptExecutor(Server(), cache_type=execution.CacheType.NONE,
            cache_args={"ram": 0, "ram_inactive": 0, "lru": 0})
        with self.assertLogs(level="ERROR"):
            executor.execute(graph, str(uuid.uuid4()), execute_outputs=["start", "end", "sink"])
        self.assertFalse(executor.success)
        errors = [message for kind, message in executor.status_messages if kind == "execution_error"]
        self.assertIn("Move the choice upstream", errors[0]["exception_message"])
        self.assertNotIn(("source", 99), EVENTS)
        self.assertFalse(loop._LIFECYCLE.runs)


if __name__ == "__main__":
    unittest.main()
