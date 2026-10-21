"""python -m unittest discover -s tests -v (FFmpeg/OpenCV/PyAV required)."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import video_privacy as vp


def frames(path):
    cap = cv2.VideoCapture(str(path))
    result = []
    while True:
        ok, image = cap.read()
        if not ok:
            break
        result.append(image)
    cap.release()
    return result


def frame_hashes(path):
    text = vp.run(['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:v:0',
                   '-f', 'framemd5', '-'])
    return [line.split(',')[-1].strip() for line in text.splitlines() if not line.startswith('#')]


class GeometryTests(unittest.TestCase):
    def test_clamping_and_transitive_union(self):
        self.assertEqual(vp.expand_box([-.1, -.1, .4, .4], 100, 100, .4), (0, 0, 46, 46))
        self.assertEqual(vp.merge_boxes([(0, 0, 10, 10), (18, 0, 10, 10), (9, 0, 10, 10)]),
                         [(0, 0, 28, 10)])

    def test_temporal_padding_multiple_faces_and_clear(self):
        args = vp.parse_args(['--margin', '0', '--hold', '.3'])
        info = {'width': 100, 'height': 100, 'fps': 30, 'duration': 2}
        entries = [{'time': .5, 'boxes': [[.1, .1, .2, .2, .9], [.7, .7, .2, .2, .9]]},
                   {'time': .7, 'boxes': [[.2, .1, .2, .2, .9]]}]
        changes = vp.timeline(entries, info, args)
        self.assertEqual(changes[0][0], .2)
        self.assertEqual(len(changes[0][1]), 2)
        self.assertEqual(changes[-1], (1.0, []))
        self.assertTrue(any(len(boxes) == 2 for _, boxes in changes))

    def test_invalid_options(self):
        for argv in (['--detect-fps', '-1'], ['--jobs', '0'], ['--duration', '0']):
            with self.assertRaises(SystemExit):
                vp.parse_args(argv)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.source = cls.root / 'input.mp4'
        vp.run(['ffmpeg', '-v', 'error', '-filter_threads', '2', '-f', 'lavfi', '-i',
                'testsrc2=size=320x240:rate=10:duration=1', '-f', 'lavfi', '-i',
                'sine=frequency=440:duration=1', '-c:v', 'libx264', '-threads', '2',
                '-crf', '0', '-c:a', 'aac', '-shortest', str(cls.source)])
        cls.info = vp.probe(cls.source)
        cls.original = frames(cls.source)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def args(self, name, *extra):
        return vp.parse_args([str(self.source), '-o', str(self.root / name),
                              '--encoder', 'cpu', '--threads', '2', '--quality', '0', *extra])

    def test_mute_keeps_video_bitstream(self):
        args = self.args('mute', '--mode', 'mute')
        output = args.output / 'input.mp4'
        vp.process_video(self.source, output, args, 'libx264')
        self.assertFalse(vp.probe(output)['audio'])
        self.assertEqual(frame_hashes(output), frame_hashes(self.source))
        before = output.stat().st_mtime_ns
        vp.process_video(self.source, output, args, 'libx264')
        self.assertEqual(output.stat().st_mtime_ns, before)

    def test_dynamic_mosaic_boundaries_and_audio_removal(self):
        args = self.args('mosaic')
        output = args.output / 'output.mp4'
        changes = [(0, [(32, 32, 96, 96), (192, 96, 64, 64)]),
                   (.4, [(96, 32, 96, 96)]), (.7, [])]
        vp.render(self.source, output, self.info, args, changes, 'libx264')
        images = frames(output)
        self.assertEqual(len(images), 10)
        self.assertFalse(vp.probe(output)['audio'])
        for n in (1, 5):
            x = 32 if n == 1 else 96
            diff = np.abs(images[n].astype(float) - self.original[n].astype(float))
            self.assertGreater(diff[40:120, x+8:x+88].mean(), 10)
            self.assertLess(diff[180:220, 20:300].mean(), 2)
        # Both boxes active initially; second box disappears when only one slot is needed.
        self.assertGreater(np.abs(images[1][104:152, 200:248].astype(float)
                                  - self.original[1][104:152, 200:248]).mean(), 10)
        for n in (7, 8, 9):
            self.assertLess(np.abs(images[n].astype(float) - self.original[n]).mean(), 2)

    def test_black_boxes_and_blur_preserves_audio(self):
        args = self.args('black', '--style', 'black', '--mode', 'blur')
        output = args.output / 'output.mp4'
        vp.render(self.source, output, self.info, args,
                  [(.2, [(32, 32, 96, 96)]), (.6, [])], 'libx264')
        images = frames(output)
        self.assertTrue(vp.probe(output)['audio'])
        self.assertLess(images[3][40:120, 40:120].mean(), 2)
        self.assertLess(np.abs(images[7].astype(float) - self.original[7]).mean(), 2)

    def test_no_faces_copies_stream_and_completed_scan_reuses_cache(self):
        args = self.args('no-faces')
        cache = self.root / 'scan-cache.json'
        cv2.setNumThreads(2)
        data = vp.scan(self.source, self.info, args, cache)
        self.assertTrue(data['complete'])
        self.assertEqual(data['detections'], [])
        self.assertEqual(data['samples'], 5)
        with patch('av.open', side_effect=AssertionError('cache should avoid decode')):
            self.assertEqual(vp.scan(self.source, self.info, args, cache), data)
        output = args.output / 'output.mp4'
        vp.render(self.source, output, self.info, args, [], 'libx264')
        self.assertEqual(frame_hashes(output), frame_hashes(self.source))

    def test_resumes_partial_scan_without_duplicate_samples(self):
        args = self.args('resume')
        cache = self.root / 'resume-cache.json'
        full = vp.scan(self.source, self.info, args, cache)
        partial = dict(full, complete=False, scanned_until=.4, next_sample=.6, samples=3)
        vp.atomic_json(cache, partial)
        resumed = vp.scan(self.source, self.info, args, cache)
        self.assertEqual(resumed['samples'], full['samples'])
        self.assertEqual(resumed['detections'], full['detections'])

    def test_all_frames_detection_and_actual_timestamps(self):
        args = self.args('all-frames', '--detect-fps', '0')
        data = vp.scan(self.source, self.info, args, self.root / 'dense-cache.json')
        self.assertEqual(data['samples'], 10)
        self.assertAlmostEqual(data['scanned_until'], .9)

    def test_vfr_timestamps_survive_masking(self):
        source = self.root / 'vfr.mp4'
        vp.run(['ffmpeg', '-v', 'error', '-i', str(self.source), '-vf',
                'select=not(eq(n\\,2)+eq(n\\,5))', '-vsync', '0', '-an',
                '-c:v', 'libx264', '-threads', '2', '-y', str(source)])
        args = self.args('vfr', '--style', 'black')
        output = args.output / 'output.mp4'
        vp.render(source, output, vp.probe(source), args,
                  [(0, [(32, 32, 64, 64)]), (.6, [])], 'libx264')
        def times(path):
            import av
            with av.open(str(path)) as container:
                return [float(frame.time) for frame in container.decode(video=0)]
        self.assertTrue(np.allclose(times(source), times(output), atol=.0001))

    def test_rotated_video_mask_coordinates(self):
        source = self.root / 'rotated.mp4'
        vp.run(['ffmpeg', '-v', 'error', '-i', str(self.source), '-c', 'copy',
                '-metadata:s:v:0', 'rotate=90', '-y', str(source)])
        info = vp.probe(source)
        self.assertEqual((info['width'], info['height']), (240, 320))
        args = self.args('rotated', '--style', 'black')
        data = vp.scan(source, info, args, self.root / 'rotated-cache.json')
        self.assertEqual(data['scan_size'], [240, 320])
        output = args.output / 'output.mp4'
        vp.render(source, output, info, args, [(0, [(32, 32, 64, 64)])], 'libx264')
        self.assertEqual(vp.probe(output)['rotation'], 0)
        image = frames(output)[0]
        self.assertEqual(image.shape[:2], (320, 240))
        self.assertLess(image[40:88, 40:88].mean(), 2)

    def test_output_conflict_is_not_overwritten(self):
        args = self.args('conflict', '--mode', 'mute')
        output = args.output / 'input.mp4'
        output.parent.mkdir()
        output.write_bytes(b'existing output')
        with self.assertRaises(ValueError):
            vp.process_video(self.source, output, args, 'libx264')
        self.assertEqual(output.read_bytes(), b'existing output')


if __name__ == '__main__':
    unittest.main()
