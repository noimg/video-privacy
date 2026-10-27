"""Adaptive sampling on timestamped clips, including interrupted two-pass scans."""
from contextlib import ExitStack
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import av
import cv2
import numpy as np

import screen_detector as sd
import video_privacy as vp


class AdaptiveScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.source = cls.root / 'appearances.mp4'
        # Red markers represent targets appearing just before a coarse probe,
        # with two far-apart appearances to exercise seeks between windows.
        with av.open(str(cls.source), 'w') as output:
            stream = output.add_stream('libx264', rate=10)
            stream.width, stream.height, stream.pix_fmt = 160, 120, 'yuv420p'
            stream.options = {'crf': '0', 'bf': '0', 'g': '10'}
            stream.codec_context.thread_count = 2
            for n in range(80):
                image = np.zeros((120, 160, 3), np.uint8)
                if 6 <= n <= 14 or 68 <= n <= 74:
                    image[:32, :32, 2] = 255
                frame = av.VideoFrame.from_ndarray(image, format='bgr24')
                frame.pts, frame.time_base = n, Fraction(1, 10)
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        cls.info = vp.probe(cls.source)
        cv2.setNumThreads(2)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def args(self, *extra):
        return vp.parse_args([str(self.source), '--targets', 'screen', '--threads', '2',
                              '--scan-mode', 'adaptive', '--refine-window', '1', *extra])

    @staticmethod
    def detect(image):
        return [[0, 0, .2, .25, .95, 'phone']] if image[8, 8, 2] > 200 else []

    def scan(self, args, cache, detect=None):
        fake = Mock(backend='cpu')
        fake.detect.side_effect = detect or self.detect
        with patch.object(sd, 'ensure_model', return_value='screen-hash'), \
                patch.object(sd, 'ScreenDetector', return_value=fake):
            data = vp.scan(self.source, self.info, args, cache)
        return data, fake

    def test_backfills_hits_reuses_probes_and_skips_gaps(self):
        args = self.args()
        data, fake = self.scan(args, self.root / 'adaptive.json')
        dense, _ = self.scan(self.args('--scan-mode', 'uniform'), self.root / 'uniform.json')
        self.assertEqual(data['detections'], dense['detections'])
        self.assertEqual([e['time'] for e in data['detections']],
                         [.6, .8, 1, 1.2, 1.4, 6.8, 7, 7.2, 7.4])
        self.assertEqual(data['refine_intervals'], [[0, 2], [6, 8]])
        self.assertEqual((data['coarse_samples'], data['refine_samples']), (8, 16))
        self.assertEqual(data['samples'], 24)
        self.assertEqual(fake.detect.call_count, 24)
        self.assertEqual(vp.timeline(data['detections'], self.info, args),
                         vp.timeline(dense['detections'], self.info, args))

    def test_empty_video_only_runs_coarse_pass(self):
        with patch('av.open', wraps=av.open) as opened:
            data, fake = self.scan(self.args(), self.root / 'empty.json', lambda _: [])
        self.assertEqual(opened.call_count, 1)
        self.assertEqual(data['samples'], 8)
        self.assertEqual(fake.detect.call_count, 8)
        self.assertEqual(data['refine_intervals'], [])
        self.assertTrue(data['complete'])
        self.assertAlmostEqual(data['scanned_until'], 7.9)

    def test_resume_each_phase_without_losing_or_duplicating_hits(self):
        args = self.args()
        expected, _ = self.scan(args, self.root / 'uninterrupted.json')
        for failed_call, phase in ((4, 'coarse'), (11, 'refine')):
            with self.subTest(phase=phase):
                cache = self.root / f'interrupted-{phase}.json'
                calls = 0

                def interrupt(image):
                    nonlocal calls
                    calls += 1
                    if calls == failed_call:
                        raise InterruptedError('test interruption')
                    return self.detect(image)

                with self.assertRaises(InterruptedError):
                    self.scan(args, cache, interrupt)
                partial = json.loads(cache.read_text())
                self.assertFalse(partial['complete'])
                self.assertEqual(partial['phase'], phase)
                resumed, _ = self.scan(args, cache)
                for key in ('detections', 'samples', 'coarse_samples', 'refine_samples', 'coarse_times'):
                    self.assertEqual(resumed[key], expected[key], key)

    def test_cache_reuse_and_adaptive_option_invalidation(self):
        cache = self.root / 'options.json'
        args = self.args()
        original, _ = self.scan(args, cache)
        with patch('av.open', side_effect=AssertionError('should reuse completed scan')):
            self.assertEqual(self.scan(args, cache)[0], original)
        args.coarse_fps = 2
        changed, _ = self.scan(args, cache)
        self.assertNotEqual(changed['signature'], original['signature'])
        self.assertEqual(changed['coarse_samples'], 16)
        args.refine_window = 2
        wider, _ = self.scan(args, cache)
        self.assertNotEqual(wider['signature'], changed['signature'])

    def test_truncated_decode_never_marks_scan_complete(self):
        cache = self.root / 'truncated.json'
        with patch.object(self, 'info', dict(self.info, duration=12)):
            with self.assertRaisesRegex(RuntimeError, '视频提前结束'):
                self.scan(self.args(), cache)
        self.assertFalse(json.loads(cache.read_text())['complete'])

    def test_adaptive_masks_video_and_keeps_uniform_cache_separate(self):
        output_dir = self.root / 'processed'
        args = self.args('-o', str(output_dir), '--encoder', 'cpu', '--style', 'black')
        output = output_dir / 'masked.mp4'
        with patch.object(sd, 'ensure_model', return_value='screen-hash'), \
                patch.object(sd, 'ScreenDetector', return_value=Mock(backend='cpu',
                                        detect=Mock(side_effect=self.detect))):
            report = vp.process_video(self.source, output, args, 'libx264')
            args.scan_mode, args.scan_only = 'uniform', True
            vp.process_video(self.source, output, args, 'libx264')
        self.assertEqual(report['render']['strategy'], 'sparse')
        self.assertGreater(report['render']['copied_frames'], 0)
        self.assertEqual((report['samples'], report['coarse_samples'], report['refine_samples']),
                         (24, 8, 16))
        self.assertEqual(report['screen_samples'], 9)
        self.assertEqual(len(list((output_dir / '.cache').glob('*.json'))), 2)
        with av.open(str(output)) as container:
            frames = [frame.to_ndarray(format='bgr24') for frame in container.decode(video=0)]
        self.assertEqual(len(frames), 80)
        for n in [6, 10, 14, 68, 70, 74]:
            self.assertLess(frames[n][8, 8].max(), 5)

    def test_cpu_face_and_cuda_screen_inference_overlap(self):
        barrier = threading.Barrier(2, timeout=5)

        def face_detect(_):
            barrier.wait()
            return None, None

        def screen_detect(_):
            barrier.wait()
            return []

        args = self.args('--scan-mode', 'uniform', '--targets', 'face', 'screen', '--duration', '1')
        with ExitStack() as stack:
            stack.enter_context(patch.object(sd, 'ensure_model', return_value='screen-hash'))
            stack.enter_context(patch.object(sd, 'ScreenDetector',
                                            return_value=Mock(backend='cuda', detect=Mock(side_effect=screen_detect))))
            stack.enter_context(patch.object(cv2.FaceDetectorYN, 'create',
                                            return_value=Mock(detect=Mock(side_effect=face_detect))))
            data = vp.scan(self.source, self.info, args, self.root / 'parallel.json')
        self.assertEqual(data['samples'], 5)

    def test_invalid_adaptive_sampling_options(self):
        for options in (['--coarse-fps', '0'], ['--coarse-fps', '6'], ['--detect-fps', '0'],
                        ['--refine-window', '-1'], ['--coarse-fps', 'nan'], ['--detect-fps', 'inf']):
            with self.subTest(options=options), self.assertRaises(SystemExit):
                self.args(*options)


if __name__ == '__main__':
    unittest.main()
