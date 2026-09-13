import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'py' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QueueClientTests(unittest.TestCase):
    def nodes(self):
        for filename, classname in (
            ('VTS_MerserkTemporalEnhance', 'VTSMerserkTemporalEnhance'),
            ('VTS_MerserkFrameInterpolate', 'VTSMerserkFrameInterpolate'),
        ):
            module = load(filename)
            yield module, getattr(module, classname)

    def test_queue_notices_extend_readiness_deadline(self):
        for module, node in self.nodes():
            messages = [dict(type='queued', position=2), dict(type='queued', position=1), dict(type='ready')]
            with self.subTest(node=node), patch.object(node, '_message', side_effect=messages) as receive, \
                 patch.object(module.time, 'monotonic', side_effect=[0, 10, 20]):
                self.assertEqual(node._ready(None, 5), dict(type='ready'))
                self.assertEqual([call.args[1] for call in receive.call_args_list], [5, 15, 25])

    def test_silent_server_still_times_out_and_bad_queue_notice_is_rejected(self):
        for module, node in self.nodes():
            with patch.object(node, '_message', side_effect=TimeoutError):
                with self.assertRaises(TimeoutError): node._ready(None, 5)
            with patch.object(node, '_message', return_value=dict(type='queued', position=False)):
                with self.assertRaises(ValueError): node._ready(None, 5)


if __name__ == '__main__': unittest.main()
