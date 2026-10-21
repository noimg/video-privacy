import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import av
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import video_privacy as vp
from sparse_render import SparseUnsupported, active_intervals


def decoded(path):
    with av.open(str(path)) as c:
        return [(f.pts * f.time_base, f.to_ndarray(format='yuv420p')) for f in c.decode(video=0)]


class SparseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.source = cls.root / 'gops.mp4'
        vp.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                'testsrc2=size=320x240:rate=10:duration=6', '-f', 'lavfi', '-i',
                'sine=frequency=440:duration=6', '-c:v', 'libx264', '-threads', '2',
                '-bf', '0', '-g', '10', '-keyint_min', '10', '-sc_threshold', '0',
                '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_primaries', 'bt709',
                '-color_trc', 'bt709', '-c:a', 'aac', '-shortest', str(cls.source)])
        cls.original = decoded(cls.source)
        cv2.setNumThreads(2)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def args(self, *options):
        return vp.parse_args([str(self.source), '--encoder', 'cpu', '--render-mode', 'sparse',
                              '--threads', '2', *options])

    def test_only_affected_gops_encoded_all_splices_decode(self):
        output = self.root / 'sparse-black.mp4'
        args = self.args('--style', 'black')
        changes = [(2.2, [(32, 32, 96, 96)]), (2.6, []),
                   (4.2, [(160, 96, 64, 64)]), (4.6, [])]
        result = vp.render(self.source, output, vp.probe(self.source), args, changes, 'libx264')
        self.assertEqual(result['strategy'], 'sparse')
        self.assertEqual(result['encoded_frames'], 20)
        self.assertEqual(result['copied_frames'], 40)
        images = decoded(output)
        self.assertEqual(len(images), 60)
        for n, ((t, original), (new_t, new)) in enumerate(zip(self.original, images)):
            self.assertEqual(t, new_t, n)
            if n < 20 or 30 <= n < 40 or n >= 50:
                self.assertTrue(np.array_equal(original, new), n)
        self.assertLess(images[23][1][40:120, 40:120].mean(), 18)
        self.assertLess(images[43][1][104:152, 168:216].mean(), 18)
        self.assertFalse(vp.probe(output)['audio'])
        with av.open(str(output)) as c:
            self.assertEqual(c.streams.video[0].codec_context.codec_tag, 'avc3')
            for f in c.decode(video=0):
                self.assertEqual(f.colorspace, 1)  # BT.709 also after encoded IDRs
        # Fail on decoder errors at every copy/encode transition.
        vp.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(output), '-map', '0:v:0',
                '-vsync', '0', '-f', 'null', '-'])

    def test_mosaic_only_changes_target_pixels_and_audio_packets_survive(self):
        output = self.root / 'sparse-mosaic.mp4'
        args = self.args('--mode', 'blur', '--quality', '1')
        changes = [(2.2, [(32, 32, 96, 96)]), (2.6, [])]
        result = vp.render(self.source, output, vp.probe(self.source), args, changes, 'libx264')
        self.assertEqual(result['encoded_frames'], 10)
        images = decoded(output)
        original, new = self.original[23][1], images[23][1]
        self.assertGreater(np.abs(original[32:128, 32:128].astype(float)
                                  - new[32:128, 32:128]).mean(), 10)
        # Decoder references must remain intact even after masking P frames.
        for n in [20, 21, 26, 27, 28, 29]:
            self.assertLess(np.abs(self.original[n][1].astype(float) - images[n][1]).mean(), .2, n)
        def audio_packets(path):
            with av.open(str(path)) as c:
                return [(p.pts * p.time_base, p.dts * p.time_base, bytes(p))
                        for p in c.demux(audio=0) if p.pts is not None]
        self.assertTrue(vp.probe(output)['audio'])
        self.assertEqual(audio_packets(self.source), audio_packets(output))

    def test_sparse_vfr_and_truncated_last_gop(self):
        source = self.root / 'sparse-vfr.mp4'
        vp.run(['ffmpeg', '-v', 'error', '-i', str(self.source), '-vf',
                'select=not(eq(n\\,22)+eq(n\\,35))', '-vsync', '0', '-an',
                '-c:v', 'libx264', '-threads', '2', '-bf', '0', '-g', '10',
                '-sc_threshold', '0', str(source)])
        args = self.args('--duration', '5.55', '--style', 'black')
        output = self.root / 'sparse-vfr-out.mp4'
        vp.render(source, output, vp.probe(source), args,
                  [(2.2, [(32, 32, 64, 64)]), (2.6, []), (5.3, [(32, 32, 64, 64)])], 'libx264')
        original = [(t, f) for t, f in decoded(source) if t < 5.55]
        images = decoded(output)
        self.assertEqual([t for t, _ in original], [t for t, _ in images])
        self.assertLess(images[-1][1][40:88, 40:88].mean(), 18)

    def test_b_frames_fallback_and_explicit_sparse_rejects(self):
        source = self.root / 'bframes.mp4'
        vp.run(['ffmpeg', '-v', 'error', '-i', str(self.source), '-t', '1', '-an',
                '-c:v', 'libx264', '-threads', '2', '-bf', '2', str(source)])
        info = vp.probe(source)
        changes = [(0, [(32, 32, 64, 64)])]
        output = self.root / 'bframes-output.mp4'
        args = self.args('--render-mode', 'auto')
        result = vp.render(source, output, info, args, changes, 'libx264')
        self.assertEqual(result['strategy'], 'full')
        before = output.read_bytes()
        with self.assertRaises(SparseUnsupported):
            vp.render(source, output, info, self.args(), changes, 'libx264')
        self.assertEqual(before, output.read_bytes())

    def test_open_gop_never_copies_unmasked_faces(self):
        args = self.args('--render-mode', 'auto')
        with patch('sparse_render.contains_idr', return_value=False):
            result = vp.render(self.source, self.root / 'fallback.mp4', vp.probe(self.source),
                               args, [(2, [(32, 32, 64, 64)]), (3, [])], 'libx264')
        self.assertEqual(result['strategy'], 'full')

    def test_lossless_profile_change_falls_back(self):
        args = self.args('--render-mode', 'auto', '--quality', '0')
        output = self.root / 'lossless-full.mp4'
        result = vp.render(self.source, output, vp.probe(self.source), args,
                           [(2.2, [(32, 32, 64, 64)]), (2.6, [])], 'libx264')
        self.assertEqual(result['strategy'], 'full')
        images = decoded(output)
        self.assertTrue(np.array_equal(self.original[20][1], images[20][1]))

    def test_full_scan_cache_survives_renderer_upgrade(self):
        args = self.args()
        # Cached detection settings deliberately keep VERSION=1 across renderer changes.
        cache = self.root / 'cached.json'
        settings = {'version': 1, 'source': vp.source_id(self.source),
                    'model': vp.MODEL_SHA256, 'fps': args.detect_fps, 'size': args.detect_size,
                    'threshold': args.threshold, 'start': 0, 'duration': None, 'rotation': 0}
        cached = {'signature': vp.digest(settings), 'complete': True, 'detections': [],
                  'settings': settings}
        vp.atomic_json(cache, cached)
        with patch('av.open', side_effect=AssertionError('should reuse old scan')):
            self.assertEqual(vp.scan(self.source, vp.probe(self.source), args, cache), cached)

    def test_timeline_ignores_faces_outside_preview(self):
        args = self.args('--duration', '2')
        info = vp.probe(self.source)
        changes = vp.timeline([{'time': 5, 'boxes': [[.1, .1, .2, .2, .9]]}], info, args)
        self.assertEqual(changes, [])
        self.assertEqual(active_intervals([(0, [(0, 0, 2, 2)]), (1, [])], 2), [(0, 1)])


if __name__ == '__main__':
    unittest.main()
