"""Expanded loops whose graph and pending expansion nodes carry identifiers only."""
import ctypes
import gc
import logging
import sys
import time
import uuid
import weakref
from contextvars import ContextVar

import psutil
import torch

import nodes
from comfy import model_management
from comfy_api.latest import io, _io
from comfy_execution.cache_provider import CacheProvider, register_cache_provider
from comfy_execution.graph import get_input_info
from comfy_execution.graph_utils import GraphBuilder, is_link


START = "VTS Memory-Aware Loop Start"
END = "VTS Memory-Aware Loop End"
VALUE = "VTS Loop Value"
COMMIT = "_VTS Memory-Aware Loop Commit"
ADVANCE = "_VTS Memory-Aware Loop Advance"
INIT = "_VTS Memory-Aware Loop Init"
ANCHOR = "_VTS Memory-Aware Loop Anchor"
INVARIANT = "_VTS Memory-Aware Loop Invariant"
FLOW = io.Custom("VTS_LOOP_FLOW")
VALUES = io.Custom("VTS_LOOP_VALUES")
_LOG = logging.getLogger(__name__)


class _Bundle:
    def __init__(self, execution, key):
        self.execution = execution
        self.key = key


class _Run:
    def __init__(self):
        self.token = uuid.uuid4().hex
        self.values = {}
        self.invariants = {}
        self.completed = {}
        self.sequence = 0

    def put(self, values):
        key = str(self.sequence)
        self.sequence += 1
        self.values[key] = values
        return key

    def get(self, key):
        if key not in self.values:
            raise ValueError("VTS loop carry has expired. Connect every loop branch to Loop End.")
        return self.values[key]

    def discard(self, key):
        self.values.pop(key, None)
        self.completed.pop(key, None)

    def bundle(self, key, final=False):
        bundle = _Bundle(self.token, key)
        if final:
            # Only a completed loop has no subsequent Commit needing this key.
            # Cached final bundles keep ownership until their consumers finish.
            weakref.finalize(bundle, self.discard, key)
        return bundle


class _LoopLifecycle(CacheProvider):
    def __init__(self):
        self.runs = {}
        self.context = ContextVar("vts_loop_execution", default=None)

    def on_prompt_start(self, prompt_id):
        previous = self.runs.pop(prompt_id, None)
        if previous is not None:
            previous.values.clear()
            previous.invariants.clear()
            previous.completed.clear()
        run = _Run()
        self.runs[prompt_id] = run
        self.context.set(run)

    def on_prompt_end(self, prompt_id):
        run = self.runs.pop(prompt_id, None)
        if run is not None:
            run.values.clear()
            run.invariants.clear()
            run.completed.clear()
        if self.context.get() is run:
            self.context.set(None)

    def current(self, token=None):
        run = self.context.get()
        if run is None or (token is not None and run.token != token):
            raise ValueError("VTS loop values belong to a different or completed execution.")
        return run

    def should_cache(self, context, value=None):
        return False

    async def on_lookup(self, context):
        return None

    async def on_store(self, context, value):
        pass


_LIFECYCLE = _LoopLifecycle()
register_cache_provider(_LIFECYCLE)


def _autogrow(name, deferred=False):
    # 100 is the framework's Autogrow limit, not a loop-specific carry limit.
    return io.Autogrow.Input(name, optional=True,
        template=io.Autogrow.TemplatePrefix(
            io.AnyType.Input("item", lazy=deferred, raw_link=deferred),
            prefix="item", min=0, max=100))


class _DeferredInputs(dict):
    """Expose Autogrow lazy slots to the scheduler's unexpanded input lookup."""
    def __contains__(self, key):
        return super().__contains__(key) or self._deferred(key)

    @staticmethod
    def _deferred(key):
        if not isinstance(key, str):
            return False
        group, separator, item = key.partition(".item")
        return bool(separator and group in ("initial", "values", "after")
                    and item.isdecimal() and str(int(item)) == item and int(item) < 100)

    def __getitem__(self, key):
        if self._deferred(key):
            return ("*", {"lazy": True, "rawLink": True})
        return super().__getitem__(key)


def _body_nodes(dynprompt, start, end):
    children = {}
    for node_id in dynprompt.all_node_ids():
        for value in dynprompt.get_node(node_id).get("inputs", {}).values():
            if is_link(value):
                children.setdefault(value[0], set()).add(node_id)
    ancestors = set()
    pending = [end]
    while pending:
        node_id = pending.pop()
        if node_id in ancestors:
            continue
        ancestors.add(node_id)
        if node_id != start:
            pending.extend(value[0] for value in dynprompt.get_node(node_id).get("inputs", {}).values()
                           if is_link(value))
    descendants = set()
    pending = [start]
    while pending:
        node_id = pending.pop()
        if node_id in descendants:
            continue
        descendants.add(node_id)
        if node_id != end:
            pending.extend(children.get(node_id, ()))
    for node_id in descendants - ancestors:
        class_type = dynprompt.get_node(node_id)["class_type"]
        if class_type == ANCHOR:
            continue
        node_class = nodes.NODE_CLASS_MAPPINGS[class_type]
        if getattr(node_class, "OUTPUT_NODE", False):
            raise ValueError("Connect loop output %s to Loop End's after inputs before continuing." % node_id)
    return ancestors & descendants


def _validate_body_inputs(dynprompt, body):
    for node_id in body:
        info = dynprompt.get_node(node_id)
        if info["class_type"] in (START, END):
            continue
        node_class = nodes.NODE_CLASS_MAPPINGS[info["class_type"]]
        schema = node_class.INPUT_TYPES()
        if issubclass(node_class, io.ComfyNode):
            schema, _, _ = _io.get_finalized_class_inputs(schema, info["inputs"])
        for name, value in info["inputs"].items():
            _, _, options = get_input_info(node_class, name, schema)
            if is_link(value) and options and options.get("lazy"):
                raise ValueError("VTS loop does not yet support connected lazy body input %s.%s. Move the choice upstream and carry the selected value." % (node_id, name))


def _expand_iteration(dynprompt, flow, end, carry):
    start = flow["start"]
    contained = _body_nodes(dynprompt, start, end)
    graph = GraphBuilder()
    clones = {}
    readers = {}
    for index, node_id in enumerate(sorted(contained)):
        clone = graph.node(dynprompt.get_node(node_id)["class_type"], "body_%d" % index)
        clone.set_override_display_id(dynprompt.get_display_node_id(node_id))
        clones[node_id] = clone
    for node_id, clone in clones.items():
        for name, value in dynprompt.get_node(node_id).get("inputs", {}).items():
            if node_id == start and (name.startswith("initial.") or name == "initial"):
                continue
            if is_link(value) and value[0] in clones:
                value = clones[value[0]].out(value[1])
            elif is_link(value) and node_id != start:
                if dynprompt.get_node(value[0])["class_type"] == INVARIANT:
                    clone.set_input(name, value)
                    continue
                source = tuple(value)
                if source not in readers:
                    readers[source] = graph.node(INVARIANT,
                        execution=flow["execution"], scope=flow["invariants"],
                        source_node=source[0], source_output=source[1])
                value = readers[source].out(0)
            clone.set_input(name, value)
    clones[start].set_input("total", flow["total"])
    clones[start].set_input("_carry", carry)
    clones[start].set_input("_iteration", flow["iteration"] + 1)
    clones[start].set_input("_invariants", flow["invariants"])
    clones[start].set_input("_owner_start", flow["owner_start"])
    return io.NodeOutput(clones[end].out(0), expand=graph.finalize())


def _cleanup(iteration, cleanup_every, collect_garbage, empty_cuda_cache, trim_cpu, log_memory):
    if cleanup_every < 1:
        raise ValueError("VTS loop cleanup_every must be at least one.")
    if (iteration + 1) % cleanup_every:
        return
    before = psutil.Process().memory_info().rss if log_memory else None
    started = time.perf_counter()
    if collect_garbage:
        gc.collect()
    if empty_cuda_cache and torch.cuda.is_initialized():
        model_management.soft_empty_cache()
    if trim_cpu:
        if sys.platform == "linux":
            libc = ctypes.CDLL(None)
            trim = getattr(libc, "malloc_trim", None)
            if trim is not None:
                trim.argtypes = [ctypes.c_size_t]
                trim.restype = ctypes.c_int
                trim(0)
            else:
                _LOG.warning("VTS loop: malloc_trim is unavailable on this system.")
        else:
            _LOG.warning("VTS loop: CPU allocator trimming is available on Linux only.")
    if log_memory:
        after = psutil.Process().memory_info().rss
        allocated = torch.cuda.memory_allocated() if torch.cuda.is_initialized() else 0
        reserved = torch.cuda.memory_reserved() if torch.cuda.is_initialized() else 0
        _LOG.info("VTS loop iteration %d: RSS %.1f -> %.1f MiB; CUDA allocated/reserved %.1f/%.1f MiB; cleanup %.3fs; active carries %d",
                  iteration + 1, before / 1048576, after / 1048576,
                  allocated / 1048576, reserved / 1048576, time.perf_counter() - started,
                  len(_LIFECYCLE.current().values))


class _ExecutionNode(io.ComfyNode):
    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return _LIFECYCLE.current().token


class VTSMemoryAwareLoopStart(_ExecutionNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=START, category="VTS/logic/loop", is_experimental=True,
            inputs=[io.Int.Input("total", default=1, min=1), _autogrow("initial", deferred=True)],
            outputs=[FLOW.Output("flow"), io.Int.Output("index"), VALUES.Output("values")],
            hidden=[io.Hidden.unique_id, io.Hidden.dynprompt], accept_all_inputs=True,
            not_idempotent=True, is_output_node=True, enable_expand=True,
            description="Carry named values through a loop. Use Loop Value to read item0, item1, etc. External linked inputs are captured once at loop entry; connected lazy body inputs are not supported.")

    @classmethod
    def INPUT_TYPES(cls):
        # Installed ComfyUI topology does not expand Autogrow before querying
        # lazy/raw flags. This local adapter is version-specific, not a core patch.
        inputs = super().INPUT_TYPES()
        inputs["optional"] = _DeferredInputs(inputs.get("optional", {}))
        return inputs

    @classmethod
    def check_lazy_status(cls, **kwargs):
        return []

    @classmethod
    def execute(cls, total, initial=None, **kwargs):
        if total < 1:
            raise ValueError("VTS loop total must be at least one.")
        run = _LIFECYCLE.current()
        carry = kwargs.get("_carry")
        if carry is None:
            dynprompt = cls.hidden.dynprompt
            start = cls.hidden.unique_id
            ends = [node_id for node_id in dynprompt.all_node_ids()
                    if dynprompt.get_node(node_id)["class_type"] == END
                    and dynprompt.get_node(node_id).get("inputs", {}).get("flow") == [start, 0]]
            if len(ends) != 1:
                raise ValueError("Connect Loop Start's flow directly to exactly one Loop End.")
            end = ends[0]
            body = _body_nodes(dynprompt, start, end)
            _validate_body_inputs(dynprompt, body)
            external = sorted({tuple(value) for node_id in body if node_id != start
                               for value in dynprompt.get_node(node_id).get("inputs", {}).values()
                               if is_link(value) and value[0] not in body})
            graph = GraphBuilder()
            init = graph.node(INIT, total=total, source_start=start, references=external)
            for name, value in (initial or {}).items():
                init.set_input("initial." + name, value)
            for index, source in enumerate(external):
                init.set_input("invariants.item%d" % index, list(source))
            # Register first-body consumers before Init completes and releases
            # its input cache. NullCache must not rerun invariant generators.
            anchor = graph.node(ANCHOR)
            dependencies = [value for name, value in dynprompt.get_node(end)["inputs"].items()
                            if name.startswith(("values.", "after.")) and is_link(value)]
            for index, value in enumerate(dependencies):
                anchor.set_input("dependencies.item%d" % index, value)
            return io.NodeOutput(init.out(0), init.out(1), init.out(2), expand=graph.finalize())
        run.get(carry)
        iteration = kwargs.get("_iteration", 0)
        flow = {"execution": run.token, "start": cls.hidden.unique_id,
                "iteration": iteration, "total": total, "carry": carry,
                "invariants": kwargs["_invariants"], "owner_start": kwargs["_owner_start"]}
        return io.NodeOutput(flow, iteration, run.bundle(carry))


class VTSMemoryAwareLoopEnd(_ExecutionNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=END, category="VTS/logic/loop", is_experimental=True,
            inputs=[FLOW.Input("flow"), _autogrow("values", deferred=True),
                _autogrow("after", deferred=True),
                io.Boolean.Input("release_previous", default=True,
                    tooltip="Release previous values from this loop's execution store. Core output caching is unchanged."),
                io.Boolean.Input("collect_garbage", default=False),
                io.Boolean.Input("empty_cuda_cache", default=False),
                io.Boolean.Input("trim_cpu", default=False),
                io.Int.Input("cleanup_every", default=1, min=1),
                io.Boolean.Input("log_memory", default=False)],
            outputs=[VALUES.Output("values")], hidden=[io.Hidden.unique_id, io.Hidden.dynprompt],
            enable_expand=True, is_output_node=True, not_idempotent=True,
            description="Update carried item values. Connect side effects to after so each finishes before the next iteration.")

    @classmethod
    def INPUT_TYPES(cls):
        # Execution's topology lookup currently does not expand V3 Autogrow slots.
        inputs = super().INPUT_TYPES()
        inputs["optional"] = _DeferredInputs(inputs.get("optional", {}))
        return inputs

    @classmethod
    def check_lazy_status(cls, **kwargs):
        return []

    @classmethod
    def execute(cls, flow, values=None, after=None, release_previous=True,
                collect_garbage=False, empty_cuda_cache=False, trim_cpu=False,
                cleanup_every=1, log_memory=False):
        _LIFECYCLE.current(flow["execution"])
        if cleanup_every < 1:
            raise ValueError("VTS loop cleanup_every must be at least one.")
        _body_nodes(cls.hidden.dynprompt, flow["start"], cls.hidden.unique_id)
        graph = GraphBuilder()
        commit = graph.node(COMMIT, flow=flow)
        for name, value in (values or {}).items():
            commit.set_input("values." + name, value)
        for name, value in (after or {}).items():
            commit.set_input("after." + name, value)
        advance = graph.node(ADVANCE, transition=commit.out(0), source_end=cls.hidden.unique_id,
            release_previous=release_previous, collect_garbage=collect_garbage,
            empty_cuda_cache=empty_cuda_cache, trim_cpu=trim_cpu,
            cleanup_every=cleanup_every, log_memory=log_memory)
        return io.NodeOutput(advance.out(0), expand=graph.finalize())


class _Commit(_ExecutionNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=COMMIT, category="VTS/logic/loop/internal", is_dev_only=True,
            inputs=[FLOW.Input("flow"), _autogrow("values"), _autogrow("after")],
            outputs=[FLOW.Output("transition")], not_idempotent=True)

    @classmethod
    def execute(cls, flow, values=None, after=None):
        run = _LIFECYCLE.current(flow["execution"])
        updated = run.get(flow["carry"]).copy()
        updated.update(values or {})
        return io.NodeOutput(dict(flow, next_carry=run.put(updated)))


class _Advance(_ExecutionNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=ADVANCE, category="VTS/logic/loop/internal", is_dev_only=True,
            inputs=[FLOW.Input("transition"), io.String.Input("source_end"),
                io.Boolean.Input("release_previous"), io.Boolean.Input("collect_garbage"),
                io.Boolean.Input("empty_cuda_cache"), io.Boolean.Input("trim_cpu"),
                io.Int.Input("cleanup_every", min=1), io.Boolean.Input("log_memory")],
            outputs=[VALUES.Output("values")], hidden=[io.Hidden.dynprompt],
            enable_expand=True, not_idempotent=True)

    @classmethod
    def execute(cls, transition, source_end, release_previous, collect_garbage,
                empty_cuda_cache, trim_cpu, cleanup_every, log_memory):
        run = _LIFECYCLE.current(transition["execution"])
        terminal = transition["iteration"] + 1 == transition["total"]
        if release_previous:
            run.discard(transition["carry"])
            # Expanded inner OUTPUT nodes can keep final bundles alive in the
            # outer expansion cache. All inner consumers have finished here.
            dynprompt = cls.hidden.dynprompt
            nested_starts = {node_id
                             for node_id in _body_nodes(dynprompt, transition["start"], source_end)
                             if node_id != transition["start"]
                             and dynprompt.get_node(node_id)["class_type"] == START}
            for key, owner in list(run.completed.items()):
                if owner in nested_starts:
                    run.discard(key)
            if terminal:
                run.invariants.pop(transition["invariants"], None)
        _cleanup(transition["iteration"], cleanup_every, collect_garbage,
                 empty_cuda_cache, trim_cpu, log_memory)
        if terminal:
            run.completed[transition["next_carry"]] = transition["owner_start"]
            return io.NodeOutput(run.bundle(transition["next_carry"], final=release_previous))
        return _expand_iteration(cls.hidden.dynprompt, transition, source_end, transition["next_carry"])


class VTSLoopValue(_ExecutionNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=VALUE, category="VTS/logic/loop", is_experimental=True,
            inputs=[VALUES.Input("values"), io.String.Input("key", default="item0")],
            outputs=[io.AnyType.Output("value")],
            description="Read one named value from a loop's values output during this execution.")

    @classmethod
    def execute(cls, values, key="item0"):
        run = _LIFECYCLE.current(values.execution)
        carried = run.get(values.key)
        if key not in carried:
            raise ValueError("VTS loop has no value named %s." % key)
        return io.NodeOutput(carried[key])


class _Init(_ExecutionNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=INIT, category="VTS/logic/loop/internal", is_dev_only=True,
            inputs=[io.Int.Input("total"), io.String.Input("source_start"),
                io.AnyType.Input("references"), _autogrow("initial"), _autogrow("invariants")],
            outputs=[FLOW.Output("flow"), io.Int.Output("index"), VALUES.Output("values")],
            not_idempotent=True)

    @classmethod
    def execute(cls, total, source_start, references, initial=None, invariants=None):
        run = _LIFECYCLE.current()
        carry = run.put(dict(initial or {}))
        scope = uuid.uuid4().hex
        run.invariants[scope] = {tuple(source): invariants["item%d" % index]
                                 for index, source in enumerate(references)}
        flow = {"execution": run.token, "start": source_start, "iteration": 0,
                "total": total, "carry": carry, "invariants": scope, "owner_start": source_start}
        return io.NodeOutput(flow, 0, run.bundle(carry))


class _Anchor(_ExecutionNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=ANCHOR, category="VTS/logic/loop/internal", is_dev_only=True,
            inputs=[io.Autogrow.Input("dependencies", optional=True,
                template=io.Autogrow.TemplatePrefix(io.AnyType.Input("item", raw_link=True),
                    prefix="item", min=0, max=100))],
            is_output_node=True, not_idempotent=True)

    @classmethod
    def execute(cls, dependencies=None):
        return io.NodeOutput()


class _Invariant(_ExecutionNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=INVARIANT, category="VTS/logic/loop/internal", is_dev_only=True,
            inputs=[io.String.Input("execution"), io.String.Input("scope"),
                io.String.Input("source_node"), io.Int.Input("source_output")],
            outputs=[io.AnyType.Output("value")])

    @classmethod
    def execute(cls, execution, scope, source_node, source_output):
        run = _LIFECYCLE.current(execution)
        return io.NodeOutput(run.invariants[scope][(source_node, source_output)])


NODE_CLASS_MAPPINGS = {START: VTSMemoryAwareLoopStart, END: VTSMemoryAwareLoopEnd,
                       VALUE: VTSLoopValue, COMMIT: _Commit, ADVANCE: _Advance,
                       INIT: _Init, ANCHOR: _Anchor, INVARIANT: _Invariant}
NODE_DISPLAY_NAME_MAPPINGS = {name: name for name in NODE_CLASS_MAPPINGS}
