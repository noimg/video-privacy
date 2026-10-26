"""Screen geometry, target selection, cache safety, and rendering integration."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import av
import cv2
import numpy as np

import screen_detector as sd
import video_privacy as vp


class ScreenGeometryTests(unittest.TestCase):
    def detector(self):
        detector = sd.ScreenDetector.__new__(sd.ScreenDetector)
        detector.threshold = .3
        detector.grids = np.zeros((8400, 2), np.float32)
        detector.strides = np.ones((8400, 1), np.float32)
        return detector

    @staticmethod
    def prediction(out, index, class_id, xywh, score=.9):
        x, y, w, h = xywh
        out[0, index, :4] = [x + w / 2, y + h / 2, np.log(w), np.log(h)]
        out[0, index, 4] = 1
        out[0, index, class_id + 5] = score

    def test_phone_laptop_monitor_nms_and_letterbox_clamping(self):
        out = np.zeros((1, 8400, 85), np.float32)
        # Source is 320x160, letterboxed at scale=2 to a 640x640 canvas.
        self.prediction(out, 0, 63, (100, 40, 400, 320))
        self.prediction(out, 1, 63, (102, 42, 400, 320), .8)  # duplicate laptop
        self.prediction(out, 2, 67, (160, 80, 60, 120))  # phone on laptop
        self.prediction(out, 3, 62, (-20, 20, 100, 100))  # clipped monitor
        self.prediction(out, 4, 67, (10, 400, 50, 50))  # padding only
        boxes = self.detector().postprocess(out, 320, 160, 2)
        self.assertEqual(sorted(b[5] for b in boxes), ['laptop', 'monitor', 'phone'])
        for x, y, w, h, _, _ in boxes:
            self.assertGreaterEqual(min(x, y, w, h), 0)
            self.assertLessEqual(x + w, 1.000001)
            self.assertLessEqual(y + h, 1.000001)
        laptop = next(b for b in boxes if b[5] == 'laptop')
        self.assertTrue(np.allclose(laptop[:4], [100 / 640, 40 / 320, 400 / 640, 280 / 320]))

    def test_other_class_and_low_confidence_do_not_mask(self):
        out = np.zeros((1, 8400, 85), np.float32)
        self.prediction(out, 0, 67, (10, 20, 100, 100), .4)
        out[0, 0, 73 + 5] = .9  # actually a book
        self.prediction(out, 1, 63, (10, 20, 100, 100), .29)
        self.assertEqual(self.detector().postprocess(out, 640, 480, 1), [])

    def test_screen_margin_and_temporal_coverage(self):
        args = vp.parse_args(['--targets', 'screen', 'face', '--margin', '.2',
                              '--screen-margin', '0', '--hold', '.1'])
        entries = [{'time': .5, 'boxes': [[.2, .2, .2, .2, .9, 'phone']], 'target': 'screen'},
                   {'time': 1.5, 'boxes': [[.2, .2, .2, .2, .9]]}]
        changes = vp.timeline(entries, {'width': 100, 'height': 100, 'fps': 30, 'duration': 2}, args)
        self.assertEqual(changes[0], (.3, [(20, 20, 20, 20)]))
        self.assertEqual(changes[1], (.7, []))
        self.assertEqual(changes[2], (1.3, [(16, 16, 28, 28)]))

    def test_preprocessing_preserves_rgb_range_and_top_left_padding(self):
        detector = self.detector()
        detector.session = Mock()
        detector.input_name = 'images'
        detector.session.run.return_value = [np.zeros((1, 8400, 85), np.float32)]
        image = np.full((160, 320, 3), [10, 20, 30], np.uint8)
        detector.detect(image)
        blob = detector.session.run.call_args.args[1]['images']
        self.assertEqual(blob.shape, (1, 3, 640, 640))
        self.assertEqual(blob.dtype, np.float32)
        self.assertTrue(np.array_equal(blob[0, :, 0, 0], [30, 20, 10]))
        self.assertTrue(np.array_equal(blob[0, :, 400, 0], [114, 114, 114]))

    def test_explicit_cuda_never_silently_falls_back(self):
        with patch('onnxruntime.get_available_providers', return_value=['CPUExecutionProvider']):
            with self.assertRaisesRegex(ValueError, 'CUDA'):
                sd.ScreenDetector(sd.MODEL, device='cuda')


class ScreenPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.source = cls.root / 'screen.mp4'
        vp.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                'testsrc2=size=320x240:rate=10:duration=1', '-c:v', 'libx264',
                '-threads', '2', '-bf', '0', '-pix_fmt', 'yuv420p', str(cls.source)])
        cls.info = vp.probe(cls.source)
        cv2.setNumThreads(2)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def args(self, *extra):
        return vp.parse_args([str(self.source), '-o', str(self.root / self._testMethodName),
                              '--targets', 'screen', '--threads', '2', '--encoder', 'cpu', *extra])

    def fake_screen_detector(self):
        detector = Mock(backend='cpu')
        detector.detect.return_value = [[.25, .25, .3, .3, .95, 'phone']]
        return detector

    def test_screen_only_scan_resume_and_cache_invalidation(self):
        args = self.args()
        cache = args.output / 'scan.json'
        fake = self.fake_screen_detector()
        with ExitStack() as stack:
            stack.enter_context(patch.object(sd, 'ensure_model', return_value='screen-hash'))
            stack.enter_context(patch.object(sd, 'ScreenDetector', return_value=fake))
            stack.enter_context(patch.object(vp, 'ensure_model', side_effect=AssertionError('face model loaded')))
            full = vp.scan(self.source, self.info, args, cache)
            self.assertEqual(full['samples'], 5)
            self.assertEqual(len(full['detections']), 5)
            with patch('av.open', side_effect=AssertionError('cache should avoid decode')):
                args.screen_margin = .2  # Rendering options must reuse scan cache.
                self.assertEqual(vp.scan(self.source, self.info, args, cache), full)
            partial = dict(full, complete=False, scanned_until=.4, next_sample=.6, samples=3,
                           detections=full['detections'][:3])
            vp.atomic_json(cache, partial)
            resumed = vp.scan(self.source, self.info, args, cache)
            self.assertEqual(resumed['detections'], full['detections'])
            self.assertEqual(resumed['samples'], full['samples'])
            fake.reset_mock()
            args.screen_threshold = .5
            rescanned = vp.scan(self.source, self.info, args, cache)
            self.assertNotEqual(rescanned['signature'], full['signature'])
            self.assertEqual(fake.detect.call_count, 5)

    def test_combined_scan_reports_faces_and_screens_and_renders_sparse(self):
        args = self.args('--targets', 'face', 'screen', '--style', 'black', '--mode', 'blur')
        output = args.output / 'masked.mp4'
        face = np.zeros((1, 15), np.float32)
        face[0, :4] = [8, 8, 16, 16]
        face[0, 14] = .99
        with patch.object(sd, 'ensure_model', return_value='screen-hash'), \
                patch.object(sd, 'ScreenDetector', return_value=self.fake_screen_detector()), \
                patch.object(cv2.FaceDetectorYN, 'create', return_value=Mock(detect=Mock(return_value=(None, face)))):
            report = vp.process_video(self.source, output, args, 'libx264')
        self.assertEqual(report['samples'], 5)
        self.assertEqual(report['face_samples'], 5)
        self.assertEqual(report['screen_samples'], 5)
        self.assertEqual(report['screen_objects'], {'monitor': 0, 'laptop': 0, 'phone': 5})
        self.assertEqual(report['render']['strategy'], 'sparse')
        with av.open(str(output)) as c:
            images = [f.to_ndarray(format='bgr24') for f in c.decode(video=0)]
        self.assertEqual(len(images), 10)
        self.assertLess(images[4][70:120, 90:150].mean(), 2)  # phone
        self.assertLess(images[4][10:20, 10:20].mean(), 2)  # face

    def test_mute_does_not_load_either_detector(self):
        args = self.args('--mode', 'mute', '--model', '/missing/face.onnx',
                         '--screen-model', '/missing/screen.onnx')
        with patch.object(sd, 'ensure_model', side_effect=AssertionError('loaded model')):
            report = vp.process_video(self.source, args.output / 'muted.mp4', args, 'libx264')
        self.assertEqual(report['render']['strategy'], 'stream-copy')
        self.assertEqual(report['screen_samples'], 0)


if __name__ == '__main__':
    unittest.main()
