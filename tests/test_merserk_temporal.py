import io
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from queue import Queue, Empty
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

spec=importlib.util.spec_from_file_location('temporal_node_test',Path(__file__).parents[1]/'py/VTS_MerserkTemporalEnhance.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class Socket:
    instances=[]
    bad_index=False
    error_after=None
    recovery_failures=0
    def __init__(self,*args,**kwargs):
        self.messages=Queue(); self.frames=[]; self.closed=False; self.header=None
        self.instances.append(self)
        self.fail_recovery = len(self.instances) <= self.recovery_failures
    def __enter__(self): return self
    def __exit__(self,*args): self.close()
    def close(self):
        self.closed=True
        self.messages.put(RuntimeError('closed'))
    def recv(self,timeout=None):
        try: result=self.messages.get(timeout=timeout)
        except Empty: raise TimeoutError()
        if isinstance(result,Exception): raise result
        return result
    def send(self,data):
        if self.closed: raise RuntimeError('closed')
        if isinstance(data,str):
            value=json.loads(data)
            if value.get('version') in (1,2):
                self.setup=value
                self.multiplier=value.get('interpolation_multiplier',1)
                self.output_count=(value['frame_count']-1)*self.multiplier+1
                self.sent=0
                self.messages.put(json.dumps(dict(type='ready',version=value['version'],chunk_bytes=module.CHUNK_BYTES,
                    output_count=self.output_count,width=value['target_width'],height=value['target_height'],channels=value['channels'])))
            elif value.get('type')=='frame':
                self.header=value; self.data=bytearray()
            elif value.get('type')=='end':
                self.messages.put(json.dumps(dict(type='done',stats={'frames':len(self.frames),'output_frames':self.sent})))
            return
        self.data.extend(data)
        if len(self.data)==self.header['bytes']:
            with Image.open(io.BytesIO(self.data)) as image:
                self.frames.append(image.copy())
                if self.fail_recovery and len(self.frames)==2:
                    self.messages.put(json.dumps(dict(type='error',message='worker died',code='NEURAL_WORKER_RESTARTED'))); return
                if self.error_after is not None and len(self.frames)>self.error_after:
                    self.messages.put(json.dumps(dict(type='error',message='GPU test failure'))); return
                with image.resize((self.setup['target_width'],self.setup['target_height'])) as result,io.BytesIO() as buffer:
                    result.save(buffer,format='PNG'); payload=buffer.getvalue()
            for _ in range(1 if self.header['index']==0 else self.multiplier):
                self.messages.put(json.dumps(dict(type='enhanced',index=99 if self.bad_index else self.sent,bytes=len(payload))))
                for offset in range(0,len(payload),module.CHUNK_BYTES):
                    self.messages.put(payload[offset:offset+module.CHUNK_BYTES])
                self.sent+=1
            if self.setup['version']==2:
                self.messages.put(json.dumps(dict(type='frame_done',index=self.header['index'])))



class TemporalNodeTests(unittest.TestCase):
    def setUp(self):
        Socket.instances=[]; Socket.bad_index=False; Socket.error_after=None; Socket.recovery_failures=0
        self.node=module.VTSMerserkTemporalEnhance()
        self.images=torch.rand(3,96,128,4)
        self.images[...,3]=180/255
        patcher=patch.object(module,'connect',Socket); patcher.start(); self.addCleanup(patcher.stop)
    def run_node(self,**kwargs):
        params=dict(enable_scaling=False,nr_passes=2,shimmer_suppression=.6)
        params.update(kwargs)
        return self.node.enhance(self.images,**params)[0]
    def test_worker_failure_replays_all_inputs_and_keeps_only_final_disk_outputs(self):
        Socket.recovery_failures=1
        with tempfile.TemporaryDirectory() as directory:
            result=self.run_node(return_type='DiskImage',output_dir=directory)
            self.assertEqual(len(Socket.instances),2)
            self.assertEqual(len(Socket.instances[-1].frames),3)
            self.assertEqual(len(list(Path(directory).iterdir())),1)
            self.assertEqual(len(list(Path(result.output_dir).glob('*.png'))),3)
            for i,frame in enumerate(Socket.instances[-1].frames):
                np.testing.assert_array_equal(np.asarray(frame),self.images[i].mul(255).round().numpy().astype(np.uint8))

    def test_second_worker_failure_stops_retrying_and_cleans_output(self):
        Socket.recovery_failures=2
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(module.MerserkWorkerRestarted):
                self.run_node(return_type='DiskImage',output_dir=directory)
            self.assertEqual(len(Socket.instances),2)
            self.assertEqual(list(Path(directory).iterdir()),[])

    def test_one_stream_ordered_rgb_alpha_both_loop_scope_and_memory(self):
        original=Image.Image.save
        def memory_only(image,target,*args,**kwargs):
            self.assertTrue(hasattr(target,'write'))
            return original(image,target,*args,**kwargs)
        with patch.object(Image.Image,'save',memory_only): result=self.run_node()
        self.assertEqual(len(Socket.instances),1)
        connection=Socket.instances[0]
        self.assertEqual(len(connection.frames),3)
        self.assertEqual(connection.setup['parameters']['nr_passes'],2)
        self.assertEqual(connection.setup['parameters']['shimmer_suppression'],.6)
        self.assertNotIn('iterations',connection.setup['parameters'])
        np.testing.assert_allclose(result.numpy(),self.images.mul(255).round().numpy()/255,atol=1e-7)
        self.assertTrue(connection.closed)
    def test_mixed_resize_and_reversed_limits(self):
        result=self.run_node(enable_scaling=True,scale_type='large',smallMaxSize=192,largeMaxSize=64)
        self.assertEqual(tuple(result.shape),(3,64,192,4))
        self.assertEqual(Socket.instances[0].frames[0].size,(128,64))
    def test_center_crop_before_upload(self):
        self.run_node(enable_scaling=True,smallMaxSize=192,largeMaxSize=192,crop='center',scale_type='large')
        self.assertEqual(Socket.instances[0].frames[0].size,(96,96))
    def test_local_downscale_and_bypass_are_offline(self):
        with patch.object(module,'connect',side_effect=AssertionError('network')):
            self.assertIs(self.run_node(enable_neural_rendering=False),self.images)
            result=self.run_node(enable_neural_rendering=False,enable_scaling=True,sizing_mode='Multiplier',upscaling_factor=.5)
        self.assertEqual(tuple(result.shape),(3,48,64,4))
    def test_scaling_only_does_not_send_neural_controls(self):
        self.run_node(enable_scaling=True,enable_neural_rendering=False,sizing_mode='Multiplier',upscaling_factor=2,nr_passes=0)
        self.assertEqual(Socket.instances[0].setup['parameters'],{})
        self.assertFalse(Socket.instances[0].setup['enable_neural_rendering'])
    def test_diskimage_input_and_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            disk=self.run_node(return_type='DiskImage',output_dir=directory)
            self.assertEqual(len(list(Path(disk.output_dir).glob('*.png'))),3)
            self.assertEqual(tuple(disk[0].shape),(96,128,4))
            result=self.node.enhance(disk,enable_scaling=False,return_type='Tensor')[0]
            self.assertEqual(tuple(result.shape),(3,96,128,4))
            same_type=self.node.enhance(disk,enable_scaling=False,output_dir=directory)[0]
            self.assertIsInstance(same_type,module.DiskImage)
    def test_bad_output_order_and_failure_remove_partial_disk_output(self):
        for bad in (False,True):
            Socket.bad_index=bad; Socket.error_after=None if bad else 1
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises((RuntimeError,ValueError)):
                    self.run_node(return_type='DiskImage',output_dir=directory)
                self.assertFalse(list(Path(directory).iterdir()))
    def test_combined_interpolation_uploads_each_source_once_and_counts_outputs(self):
        for multiplier in (2,3,4,8):
            result=self.run_node(enable_frame_interpolation=True,interpolation_multiplier=multiplier)
            self.assertEqual(tuple(result.shape),(2*multiplier+1,96,128,4))
            connection=Socket.instances[-1]
            self.assertEqual(len(connection.frames),3)
            self.assertEqual(connection.setup['version'],2)
            self.assertEqual(connection.setup['interpolation_multiplier'],multiplier)

    def test_interpolation_only_and_downscale_still_contact_server(self):
        result=self.run_node(enable_frame_interpolation=True,enable_neural_rendering=False,
            enable_scaling=True,sizing_mode='Multiplier',upscaling_factor=.5)
        self.assertEqual(tuple(result.shape),(5,48,64,4))
        self.assertEqual(Socket.instances[-1].setup['parameters'],{})
        self.assertEqual(Socket.instances[-1].frames[0].size,(64,48))

    def test_single_frame_skips_interpolation_and_disabled_options_are_ignored(self):
        self.images=self.images[:1]
        self.assertIs(self.run_node(enable_frame_interpolation=True,enable_neural_rendering=False),self.images)
        self.assertFalse(Socket.instances)
        self.run_node(interpolation_multiplier=99)
        self.assertEqual(Socket.instances[-1].setup['version'],1)

    def test_combined_recovery_keeps_only_complete_lossless_webp_output(self):
        Socket.recovery_failures=1
        with tempfile.TemporaryDirectory() as directory:
            result=self.run_node(enable_frame_interpolation=True,interpolation_multiplier=4,
                return_type='DiskImage',output_dir=directory,format='webp',prefix='enhanced',start_sequence=100)
            self.assertEqual(len(Socket.instances),2)
            self.assertEqual(len(result),9)
            self.assertEqual(len(list(Path(directory).iterdir())),1)
            self.assertTrue((Path(result.output_dir)/'enhanced_000100.webp').is_file())
            self.assertEqual(len(list(Path(result.output_dir).glob('*.webp'))),9)
            np.testing.assert_allclose(result[0].numpy(), self.images[0].mul(255).round().numpy()/255,atol=1e-7)

    def test_output_names_and_formats_cannot_escape_output_folder(self):
        for options in (dict(prefix='../escape'),dict(prefix='a/b'),dict(prefix='a\\b'),dict(format='../png')):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    self.run_node(return_type='DiskImage',output_dir=directory,**options)
                self.assertFalse(list(Path(directory).iterdir()))

    def test_jpeg_disk_output_quality_rgb_metadata_and_png_transport(self):
        self.images=torch.zeros(3,96,128,4)
        self.images[...,0]=1  # Transparent red should become white in JPEG.
        original=Image.Image.save
        qualities=[]
        def observe(image,target,*args,**kwargs):
            if isinstance(target,Path) and target.suffix=='.jpg':
                qualities.append(kwargs.get('quality'))
            else:
                self.assertEqual(kwargs.get('format'),'PNG')
            return original(image,target,*args,**kwargs)
        with tempfile.TemporaryDirectory() as directory,patch.object(Image.Image,'save',observe):
            result=self.run_node(enable_frame_interpolation=True,return_type='DiskImage',
                output_dir=directory,format='jpg',quality=87)
            self.assertEqual(result.shape,(5,96,128,3))
            self.assertEqual(tuple(result[0].shape),(96,128,3))
            self.assertTrue(torch.all(result[0]>.99))
            self.assertEqual(qualities,[87]*5)
            for path in Path(result.output_dir).glob('*.jpg'):
                with Image.open(path) as image:
                    self.assertEqual((image.format,image.mode),('JPEG','RGB'))
            self.assertEqual(Socket.instances[-1].frames[0].mode,'RGBA')
            self.assertNotIn('quality',Socket.instances[-1].setup)

    def test_jpeg_format_converts_disk_input_even_without_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            source=self.run_node(return_type='DiskImage',output_dir=directory)
            with patch.object(module,'connect',side_effect=AssertionError('network')):
                result=self.node.enhance(source,enable_scaling=False,enable_neural_rendering=False,
                    return_type='DiskImage',output_dir=directory,format='jpg')[0]
            self.assertIsNot(result,source)
            self.assertEqual(result.format,'jpg')
            self.assertEqual(result.shape[-1],3)
            self.assertEqual(len(list(Path(result.output_dir).glob('*.jpg'))),3)

    def test_input_controls_have_tooltips_without_outer_iterations_or_hdr(self):
        schema=self.node.INPUT_TYPES(); fields=schema['required']|schema['optional']
        self.assertNotIn('iterations',fields)
        self.assertNotIn('hdr_mode',fields)
        self.assertIn('shimmer_suppression',fields)
        for name,value in fields.items(): self.assertTrue(value[1].get('tooltip'),name)


if __name__=='__main__': unittest.main()
