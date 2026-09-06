"""Independent CPU scheduler regressions for nested and shared loop ownership."""
import unittest
from unittest.mock import patch

import test_memory_aware_loop_scheduler as fixture


class IndependentLoopTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.SchedulerTests()
        self.fixture.setUp()

    def test_twenty_outer_three_inner_iterations_have_bounded_vts_ownership(self):
        for mode in (fixture.execution.CacheType.NONE, fixture.execution.CacheType.RAM_PRESSURE):
            with self.subTest(mode=mode):
                self.fixture.setUp()
                graph = fixture.prompt(20)
                graph["inner_start"] = fixture.node(fixture.loop.START, total=3,
                    **{"initial.item0": ["read", 0]})
                graph["inner_read"] = fixture.node(fixture.loop.VALUE,
                    values=["inner_start", 2], key="item0")
                graph["body"]["inputs"]["value"] = ["inner_read", 0]
                graph["inner_end"] = fixture.node(fixture.loop.END, flow=["inner_start", 0],
                    **{"values.item0": ["body", 0]})
                graph["inner_final"] = fixture.node(fixture.loop.VALUE,
                    values=["inner_end", 0], key="item0")
                graph["end"]["inputs"]["values.item0"] = ["inner_final", 0]
                counts = []
                original_execute = fixture.Body.execute

                def record_ownership(body, value, offset):
                    fixture.gc.collect()
                    run = fixture.loop._LIFECYCLE.current()
                    counts.append((len(run.values), len(run.invariants), len(run.completed)))
                    return original_execute(body, value, offset)

                with patch.object(fixture.Body, "execute", record_ownership):
                    self.fixture.run_prompt(graph, mode)
                self.assertEqual(fixture.RESULTS, [61])
                self.assertEqual(sum(event[0] == "source" for event in fixture.EVENTS), 1)
                self.assertEqual(len(counts), 60)
                self.assertLessEqual(max(count[0] for count in counts), 2)
                self.assertLessEqual(max(count[1] for count in counts), 2)
                self.assertEqual(max(count[2] for count in counts), 0)
                if mode == fixture.execution.CacheType.NONE:
                    self.assertLessEqual(max(fixture.ALIVE), 2)

    def test_independent_loops_share_one_external_source_execution(self):
        for mode in (fixture.execution.CacheType.NONE, fixture.execution.CacheType.RAM_PRESSURE):
            with self.subTest(mode=mode):
                self.fixture.setUp()
                graph = fixture.prompt(4)
                for key, node in fixture.prompt(3).items():
                    if key == "source":
                        continue
                    for name, value in node["inputs"].items():
                        if fixture.loop.is_link(value) and value[0] != "source":
                            node["inputs"][name] = ["second_" + value[0], value[1]]
                    graph["second_" + key] = node
                self.fixture.run_prompt(graph, mode)
                self.assertEqual(sorted(fixture.RESULTS), [4, 5])
                self.assertEqual(sum(event[0] == "source" for event in fixture.EVENTS), 1)
                self.assertEqual(sum(event[0] == "body" for event in fixture.EVENTS), 7)


if __name__ == "__main__":
    unittest.main()
