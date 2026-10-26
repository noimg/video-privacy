#!/usr/bin/env python3
"""批量静音、人脸与手机/电脑屏幕打码。运行 python video_privacy.py --help。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction

import screen_detector

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'models/face_detection_yunet_2023mar.onnx'
MODEL_SHA256 = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'
MODEL_URL = ('https://media.githubusercontent.com/media/opencv/opencv_zoo/main/'
             'models/face_detection_yunet/face_detection_yunet_2023mar.onnx')
EXTENSIONS = {'.mp4', '.mov', '.mkv', '.avi', '.m4v', '.webm', '.mts', '.m2ts', '.ts'}
VERSION = 1
RENDER_VERSION = 2


def log(message):
    print(message, flush=True)


def run(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        raise RuntimeError(f"命令失败：{command[0]}\n{result.stderr[-5000:]}")
    return result.stdout


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_id(path):
    stat = path.stat()
    return {'path': str(path.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def probe(path):
    data = json.loads(run(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                           '-of', 'json', str(path)]))
    streams = [s for s in data['streams'] if s['codec_type'] == 'video'
               and not s.get('disposition', {}).get('attached_pic')]
    if not streams:
        raise ValueError(f'没有视频轨：{path}')
    v = streams[0]
    rotation = 0
    for side in v.get('side_data_list', []):
        if 'rotation' in side:
            rotation = int(round(float(side['rotation']))) % 360
            break
    else:
        rotation = -int(v.get('tags', {}).get('rotate', 0)) % 360
    if rotation not in (0, 90, 180, 270):
        raise ValueError(f'暂不支持 {rotation} 度旋转：{path}')
    w, h = v['width'], v['height']
    if rotation in (90, 270):
        w, h = h, w
    duration = float(v.get('duration') or data['format'].get('duration', 0))
    return {'width': w, 'height': h, 'rotation': rotation, 'duration': duration,
            'fps': float(Fraction(v.get('avg_frame_rate', '0/1'))),
            'video_index': v['index'], 'codec': v['codec_name'],
            'audio': any(s['codec_type'] == 'audio' for s in data['streams'])}


def ensure_model(path):
    if not path.exists():
        if path.resolve() != MODEL:
            raise ValueError(f'找不到模型：{path}')
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        log('下载 YuNet 人脸检测模型（约 228 KB）…')
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=30) as response:
                with temporary.open('wb') as output:
                    shutil.copyfileobj(response, output)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != MODEL_SHA256:
                raise ValueError('模型校验失败')
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.resolve() == MODEL and sha != MODEL_SHA256:
        raise ValueError('内置模型校验失败，请重新下载模型')
    return sha


def scan(path, info, args, cache_path):
    import av
    import cv2
    import numpy as np

    # Reuse libswscale's context and bound its pool. Frame.reformat() creates a
    # fresh context for each sampled frame and otherwise defaults to all CPUs.
    reformatter = av.video.reformatter.VideoReformatter()
    faces_enabled = 'face' in args.targets
    screens_enabled = 'screen' in args.targets
    model_sha = ensure_model(args.model) if faces_enabled else None
    settings = {'version': VERSION, 'source': source_id(path), 'model': model_sha,
                'fps': args.detect_fps, 'size': args.detect_size, 'threshold': args.threshold,
                'start': args.start, 'duration': args.duration, 'rotation': info['rotation']}
    # Keep face-only signatures compatible with previously saved long scans.
    if screens_enabled:
        settings['screens'] = {'version': screen_detector.VERSION, 'targets': args.targets,
                               'model': screen_detector.ensure_model(args.screen_model),
                               'threshold': args.screen_threshold, 'device': args.screen_device}
    signature = digest(settings)
    cached = {}
    if cache_path.exists() and not args.rescan:
        cached = json.loads(cache_path.read_text(encoding='utf-8'))
        if cached.get('signature') != signature:
            cached = {}
    if cached.get('complete'):
        log(f'[{path.name}] 使用检测缓存，{len(cached["detections"])} 组检测结果')
        return cached

    size = max(args.detect_size if faces_enabled else 0, 640 if screens_enabled else 0)
    scale = min(1, size / max(info['width'], info['height']))
    w, h = max(32, round(info['width'] * scale)), max(32, round(info['height'] * scale))
    face_scale = min(1, args.detect_size / max(info['width'], info['height']))
    fw, fh = max(32, round(info['width'] * face_scale)), max(32, round(info['height'] * face_scale))
    detector = (cv2.FaceDetectorYN.create(str(args.model), '', (fw, fh), args.threshold, .3, 5000)
                if faces_enabled else None)
    screens = (screen_detector.ScreenDetector(args.screen_model, args.screen_threshold,
                                              args.screen_device, args.threads)
               if screens_enabled else None)
    total = min(info['duration'] - args.start, args.duration or math.inf)
    if total <= 0:
        raise ValueError('起始时间超出视频长度')
    data = cached or {'signature': signature, 'settings': settings, 'detections': [],
                      'scanned_until': -1.0, 'next_sample': 0.0, 'samples': 0,
                      'complete': False, 'scan_seconds': 0.0}
    if screens:
        data['screen_backend'] = screens.backend
        log(f'[{path.name}] 屏幕检测：YOLOX-S / {screens.backend.upper()}，手机、笔记本、显示器')
    interval = 1 / args.detect_fps if args.detect_fps else 0
    began = time.monotonic()
    last_save = last_log = began
    elapsed_before = data['scan_seconds']
    resume = data['scanned_until']
    log(f'[{path.name}] 扫描 {total:.1f}s，检测尺寸 {w}×{h}，'
        f'{str(args.detect_fps) + " fps" if args.detect_fps else "逐帧"}'
        + (f'，从 {resume:.1f}s 继续' if resume > 0 else ''))

    def save():
        data['scan_seconds'] = elapsed_before + time.monotonic() - began
        atomic_json(cache_path, data)

    try:
        with av.open(str(path)) as container:
            stream = next(s for s in container.streams.video if s.index == info['video_index'])
            stream.thread_type = 'AUTO'
            stream.codec_context.thread_count = args.threads
            origin = float((stream.start_time or 0) * stream.time_base)
            seek_to = args.start + max(0, resume)
            if seek_to > 0:
                container.seek(int((seek_to + origin) / stream.time_base), stream=stream,
                               backward=True, any_frame=False)
            for frame in container.decode(stream):
                if STOP.is_set():
                    raise InterruptedError('已停止，检测进度已保存')
                if frame.pts is None:
                    raise ValueError('视频帧缺少时间戳，无法准确定位打码位置')
                t = float(frame.pts * frame.time_base) - origin - args.start
                if t < 0 or t <= resume:
                    continue
                if t >= total:
                    break
                if interval and t + 1e-6 < data['next_sample']:
                    data['scanned_until'] = t
                    continue
                rw, rh = (h, w) if info['rotation'] in (90, 270) else (w, h)
                image = reformatter.reformat(frame, width=rw, height=rh, format='bgr24',
                                             threads=args.threads).to_ndarray()
                if info['rotation']:
                    image = np.ascontiguousarray(np.rot90(image, info['rotation'] // 90))
                faces = None
                if detector:
                    face_image = cv2.resize(image, (fw, fh)) if (fw, fh) != (w, h) else image
                    _, faces = detector.detect(face_image)
                screen_boxes = screens.detect(image) if screens else []
                data['samples'] += 1
                if faces is not None:
                    boxes = []
                    for f in faces:
                        x, y, bw, bh = map(float, f[:4])
                        boxes.append([x / fw, y / fh, bw / fw, bh / fh, float(f[14])])
                    data['detections'].append({'time': round(t, 6), 'boxes': boxes})
                if screen_boxes:
                    data['detections'].append({'time': round(t, 6), 'boxes': screen_boxes,
                                               'target': 'screen'})
                if interval:
                    data['next_sample'] = (math.floor((t + 1e-6) / interval) + 1) * interval
                data['scanned_until'] = t
                now = time.monotonic()
                if now - last_save >= 30:
                    save()
                    last_save = now
                if now - last_log >= 15:
                    speed = (t - max(0, resume)) / max(.001, now - began)
                    log(f'[{path.name}] 扫描 {t:.0f}/{total:.0f}s，{speed:.1f} 倍速，'
                        f'检测结果 {len(data["detections"])} 组')
                    last_log = now
        # A truncated decoder must never turn an unscanned tail into "no targets".
        if data['scanned_until'] < total - max(1.0, 3 / max(info['fps'], 1)):
            raise RuntimeError(f'视频提前结束：仅扫描到 {data["scanned_until"]:.2f}/{total:.2f}s')
        data['complete'] = True
        data['scan_size'] = [w, h]
        save()
    except BaseException:
        save()
        raise
    return data


def expand_box(box, width, height, margin):
    x, y, w, h = box[:4]
    x0 = max(0, math.floor((x - w * margin) * width / 2) * 2)
    y0 = max(0, math.floor((y - h * margin) * height / 2) * 2)
    x1 = min(width, math.ceil((x + w * (1 + margin)) * width / 2) * 2)
    y1 = min(height, math.ceil((y + h * (1 + margin)) * height / 2) * 2)
    return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))


def merge_boxes(boxes):
    """Union intersecting masks, including masks from adjacent sample times."""
    result = []
    for box in sorted(set(boxes)):
        x, y, w, h = box
        if w <= 0 or h <= 0:
            continue
        while True:
            for i, (a, b, c, d) in enumerate(result):
                if x <= a + c and a <= x + w and y <= b + d and b <= y + h:
                    x1, y1 = max(x + w, a + c), max(y + h, b + d)
                    x, y = min(x, a), min(y, b)
                    w, h = x1 - x, y1 - y
                    result.pop(i)
                    break
            else:
                break
        result.append((x, y, w, h))
    return sorted(result)


def timeline(detections, info, args):
    """Sweep temporal events; retain neighboring boxes for movement and boundary coverage."""
    events = []
    guard = max(args.hold, 1 / args.detect_fps if args.detect_fps else 1 / max(info['fps'], 1))
    length = min(info['duration'] - args.start, args.duration or math.inf)
    for i, entry in enumerate(detections):
        if entry['time'] - guard >= length or entry['time'] + guard <= 0:
            continue
        margin = args.screen_margin if entry.get('target') == 'screen' else args.margin
        boxes = [expand_box(b, info['width'], info['height'], margin) for b in entry['boxes']]
        events.extend([(round(max(0, entry['time'] - guard), 6), 1, i, boxes),
                       (round(min(length, entry['time'] + guard), 6), -1, i, [])])
    events.sort(key=lambda e: e[0])
    active = {}
    changes = []
    previous = []
    p = 0
    while p < len(events):
        t = events[p][0]
        while p < len(events) and events[p][0] == t:
            _, action, i, boxes = events[p]
            if action == 1:
                active[i] = boxes
            else:
                active.pop(i, None)
            p += 1
        boxes = merge_boxes(b for group in active.values() for b in group)
        if boxes != previous:
            changes.append((t, boxes))
            previous = boxes
    return changes


def write_commands(changes, path, overlay_control=False):
    slots = max((len(boxes) for _, boxes in changes), default=0)
    old = [None] * slots
    overlay_enabled = False
    with path.open('w', encoding='utf-8') as out:
        for t, boxes in changes:
            commands = []
            if overlay_control and bool(boxes) != overlay_enabled:
                overlay_enabled = bool(boxes)
                commands.append(f'overlay@privacy enable {int(overlay_enabled)}')
            for i in range(slots):
                box = boxes[i] if i < len(boxes) else None
                if box == old[i]:
                    continue
                target = f'drawbox@face{i}'
                if box is None:
                    commands.append(f'{target} enable 0')
                else:
                    commands.extend(f'{target} {key} {value}' for key, value in zip('xywh', box))
                    commands.append(f'{target} enable 1')
                old[i] = box
            if commands:
                out.write(f'{t:.6f} ' + ', '.join(commands) + ';\n')
    return slots


def filter_graph(info, args, slots):
    # Commands live in the working directory; no interpolation of user paths into lavfi syntax.
    color = 'white' if args.style == 'mosaic' else 'black'
    boxes = ','.join(f'drawbox@face{i}=x=0:y=0:w=2:h=2:t=fill:c={color}:enable=0'
                     for i in range(slots))
    control = f'sendcmd=f=commands.txt,{boxes}'
    prefix = (f'[0:{info["video_index"]}]setpts=PTS-STARTPTS,'
              'pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p')
    if args.style == 'black':
        return f'{prefix},{control}[out]'
    w, h = math.ceil(info['width'] / 2) * 2, math.ceil(info['height'] / 2) * 2
    pw, ph = max(1, math.ceil(w / args.block_size)), max(1, math.ceil(h / args.block_size))
    # Draw and combine alpha at the mosaic grid size, avoiding full-resolution LUTs
    # and repeated gray/YUV conversions. White in limited-range YUV has luma 235.
    return (f'{prefix},split=2[base][small];'
            f'[small]scale={pw}:{ph}:flags=area,split=2[pixels][masksrc];'
            f'[masksrc]lut=y=0,{control},extractplanes=y,lut=y=val*255/235[mask];'
            f'[pixels][mask]alphamerge,scale={w}:{h}:flags=neighbor[covered];'
            '[base][covered]overlay@privacy=0:0:format=yuv420:enable=0[out]')


def grid_timeline(changes, info, block_size):
    w, h = math.ceil(info['width'] / 2) * 2, math.ceil(info['height'] / 2) * 2
    pw, ph = math.ceil(w / block_size), math.ceil(h / block_size)
    result = []
    previous = None
    for t, boxes in changes:
        cells = []
        for x, y, bw, bh in boxes:
            x0, y0 = math.floor(x * pw / w), math.floor(y * ph / h)
            x1, y1 = math.ceil((x + bw) * pw / w), math.ceil((y + bh) * ph / h)
            cells.append((x0, y0, x1 - x0, y1 - y0))
        cells = merge_boxes(cells)
        if cells != previous:
            result.append((t, cells))
            previous = cells
    return result


def choose_encoder(requested):
    if requested == 'cpu':
        return 'libx264'
    command = ['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=s=128x128:d=0.1',
               '-c:v', 'h264_nvenc', '-f', 'null', '-']
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        log('编码器：NVIDIA NVENC')
        return 'h264_nvenc'
    if requested == 'nvenc':
        raise RuntimeError('NVENC 实测不可用：\n' + result.stderr[-2000:])
    log('NVENC 实测不可用，自动使用 CPU / libx264。')
    return 'libx264'


def render(path, output, info, args, changes, encoder):
    if not changes or args.render_mode == 'full':
        return render_full(path, output, info, args, changes, encoder)
    from sparse_render import SparseUnsupported, render_sparse
    if encoder != 'libx264':
        if args.render_mode == 'sparse':
            raise ValueError('--render-mode sparse 目前需要 --encoder cpu')
        return render_full(path, output, info, args, changes, encoder)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix='.privacy-', dir=output.parent) as tmp:
            temporary = Path(tmp) / 'sparse.mp4'
            result = render_sparse(path, temporary, info, args, changes, STOP, log)
            checked = probe(temporary)
            if args.mode != 'blur' and checked['audio']:
                raise RuntimeError('输出验证失败：仍有音频轨')
            if args.mode == 'blur' and info['audio'] and not checked['audio']:
                raise RuntimeError('输出验证失败：音频轨丢失')
            if (checked['width'], checked['height']) != (info['width'], info['height']):
                raise RuntimeError('输出验证失败：分辨率改变')
            expected = min(info['duration'], args.duration or math.inf)
            if abs(checked['duration'] - expected) > 1:
                raise RuntimeError('分段输出时长验证失败')
            result['output'] = checked
            os.replace(temporary, output)
            return result
    except SparseUnsupported as exc:
        if args.render_mode == 'sparse':
            raise
        log(f'[{path.name}] 使用完整编码：{exc}')
        return render_full(path, output, info, args, changes, encoder)


def render_full(path, output, info, args, changes, encoder):
    began = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.privacy-', dir=output.parent) as tmp:
        work = Path(tmp)
        temporary = work / 'output.mp4'
        base = ['ffmpeg', '-hide_banner', '-nostdin', '-y', '-loglevel', 'warning',
                '-threads', str(args.threads)]
        if args.start:
            base += ['-ss', str(args.start)]
        base += ['-i', str(path)]
        if args.duration:
            base += ['-t', str(args.duration)]
        if changes:
            commands = (grid_timeline(changes, info, args.block_size)
                        if args.style == 'mosaic' else changes)
            slots = write_commands(commands, work / 'commands.txt', args.style == 'mosaic')
            (work / 'filters.txt').write_text(filter_graph(info, args, slots), encoding='utf-8')
            base += ['-filter_complex_threads', str(args.threads), '-filter_complex_script',
                     'filters.txt', '-map', '[out]', '-c:v', encoder]
            if encoder == 'h264_nvenc':
                base += ['-preset', 'p4', '-rc', 'vbr', '-cq', str(args.quality), '-b:v', '0']
            else:
                base += ['-preset', args.preset, '-crf', str(args.quality)]
            base += ['-threads', str(args.threads), '-vsync', '0', '-enc_time_base', '-1',
                     '-metadata:s:v:0', 'rotate=0']
            if args.mode == 'blur' and info['audio']:
                base += ['-map', '0:a?', '-c:a', 'aac', '-b:a', '192k']
            else:
                base += ['-an']
        else:
            # Audio removal and videos with no detections never pass through an encoder.
            base += ['-map', f'0:{info["video_index"]}', '-c:v', 'copy']
            if args.mode == 'blur' and info['audio']:
                base += ['-map', '0:a?', '-c:a', 'aac', '-b:a', '192k']
            else:
                base += ['-an']
        base += ['-map_metadata', '-1', '-map_chapters', '-1', '-sn', '-dn',
                 '-movflags', '+faststart', '-progress', 'pipe:1', '-nostats', str(temporary)]
        log(f'[{path.name}] ' + ('开始打码编码' if changes else '复制视频流，无需重编码'))
        with (work / 'ffmpeg.log').open('w') as errors:
            process = subprocess.Popen(base, cwd=work, stdout=subprocess.PIPE, stderr=errors,
                                       text=True)
            last_log = time.monotonic()
            try:
                for line in process.stdout:
                    if STOP.is_set():
                        raise InterruptedError('已停止输出，可再次运行继续')
                    if line.startswith('out_time_ms=') and time.monotonic() - last_log > 15:
                        try:
                            t = int(line.strip().split('=')[1]) / 1_000_000
                        except ValueError:
                            continue
                        log(f'[{path.name}] 输出 {t:.0f}s，'
                            f'{t / max(.001, time.monotonic() - began):.1f} 倍速')
                        last_log = time.monotonic()
                if process.wait():
                    raise RuntimeError((work / 'ffmpeg.log').read_text()[-5000:])
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                process.stdout.close()
            errors.flush()
        error_log = (work / 'ffmpeg.log').read_text()
        if 'Failed to process command' in error_log or 'Function not implemented' in error_log:
            raise RuntimeError('FFmpeg 不支持动态打码命令：\n' + error_log[-5000:])
        checked = probe(temporary)
        if args.mode != 'blur' and checked['audio']:
            raise RuntimeError('输出验证失败：仍有音频轨')
        if changes and (checked['width'], checked['height']) != (
                math.ceil(info['width'] / 2) * 2, math.ceil(info['height'] / 2) * 2):
            raise RuntimeError('输出分辨率验证失败')
        expected = min(info['duration'] - args.start, args.duration or math.inf)
        # Stream copying a clip can retain GOP preroll; all other outputs must match closely.
        tolerance = 10 if not changes and (args.start or args.duration) else 1
        if abs(checked['duration'] - expected) > tolerance:
            raise RuntimeError(f'输出时长异常：{checked["duration"]:.3f}s，预期 {expected:.3f}s')
        os.replace(temporary, output)
    return {'seconds': time.monotonic() - began, 'output': checked,
            'strategy': 'full' if changes else 'stream-copy',
            'encoder': encoder if changes else 'stream-copy'}


def process_video(path, output, args, encoder):
    began = time.monotonic()
    info = probe(path)
    if args.start >= info['duration']:
        raise ValueError('起始时间超出视频长度')
    ignored = {'input', 'output', 'jobs', 'overwrite', 'rescan', 'scan_only'}
    if args.mode == 'mute' or args.targets == ['face']:
        ignored.update({'targets', 'screen_model', 'screen_threshold', 'screen_device', 'screen_margin'})
    identity = {'version': RENDER_VERSION, 'source': source_id(path),
                'options': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                            if k not in ignored},
                'model_sha256': (ensure_model(args.model)
                                 if args.mode != 'mute' and 'face' in args.targets else None)}
    if args.mode != 'mute' and 'screen' in args.targets:
        identity['screen_model_sha256'] = screen_detector.ensure_model(args.screen_model)
    signature = digest(identity)
    report_path = output.with_suffix('.report.json')
    if output.exists() and not args.scan_only:
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        if not args.overwrite:
            if (report.get('signature') == signature
                    and report.get('output_size') == output.stat().st_size):
                log(f'[{path.name}] 已完成，跳过（--overwrite 可重做）')
                return report
            raise ValueError(f'输出已存在且参数或来源不同：{output}。使用 --overwrite 覆盖。')
    cache_name = digest(str(path))[:20]
    if 'screen' in args.targets:
        cache_name += '-' + '-'.join(args.targets)
    cache_path = args.output / '.cache' / (cache_name + '.json')
    data = None if args.mode == 'mute' else scan(path, info, args, cache_path)
    if args.scan_only:
        log(f'[{path.name}] 检测已保存：{cache_path}')
        return {'input': str(path), 'cache': str(cache_path)}
    changes = timeline(data['detections'], info, args) if data else []
    render_result = render(path, output, info, args, changes, encoder)
    report = {'signature': signature, 'input': str(path), 'output': str(output),
              'source': info, 'mode': args.mode, 'style': args.style,
              'targets': args.targets, 'detect_fps': args.detect_fps,
              'face_samples': sum(e.get('target', 'face') == 'face'
                                  for e in data['detections']) if data else 0,
              'screen_samples': sum(e.get('target') == 'screen'
                                    for e in data['detections']) if data else 0,
              'screen_backend': data.get('screen_backend') if data else None,
              'screen_objects': {name: sum(b[5] == name for e in data['detections']
                                          if e.get('target') == 'screen' for b in e['boxes'])
                                 if data else 0 for name in screen_detector.CLASSES.values()},
              'samples': data['samples'] if data else 0,
              'scan_seconds': data['scan_seconds'] if data else 0,
              'render': render_result, 'total_seconds': time.monotonic() - began,
              'output_size': output.stat().st_size}
    atomic_json(report_path, report)
    log(f'[{path.name}] 完成：{output}，本次耗时 {report["total_seconds"]:.1f}s')
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='批量去音频、人脸及手机/电脑屏幕打码；默认静音+人脸打码。',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', nargs='?', type=Path, default=ROOT.parent / 'Videos-origin',
                        help='输入文件或目录，目录递归处理')
    parser.add_argument('-o', '--output', type=Path, default=ROOT.parent / 'Videos-processed',
                        help='输出目录（保留子目录结构）')
    parser.add_argument('--mode', choices=['both', 'mute', 'blur'], default='both',
                        help='both=静音+打码，mute=仅静音，blur=仅打码并保留音频')
    parser.add_argument('--style', choices=['mosaic', 'black'], default='mosaic', help='打码样式')
    parser.add_argument('--targets', nargs='+', choices=['face', 'screen'], default=['face'],
                        help='打码对象：face=人脸，screen=手机/笔记本/显示器；可同时指定')
    parser.add_argument('--detect-fps', type=float, default=5,
                        help='人脸和屏幕每秒检测帧数；0=逐帧检测，抽帧可能漏掉短暂目标')
    parser.add_argument('--detect-size', type=int, default=640,
                        help='人脸检测图最长边；小脸多时可设 960/1280；屏幕模型固定输入 640×640')
    parser.add_argument('--threshold', type=float, default=.65, help='人脸置信度阈值')
    parser.add_argument('--screen-model', type=Path, default=screen_detector.MODEL,
                        help='YOLOX-S 640 COCO ONNX 模型路径，默认模型缺失时自动下载')
    parser.add_argument('--screen-threshold', type=float, default=.3, help='屏幕设备置信度阈值')
    parser.add_argument('--screen-device', choices=['auto', 'cpu', 'cuda'], default='auto',
                        help='屏幕推理设备；auto 优先使用 ONNX Runtime CUDA，未安装 GPU 版时使用 CPU')
    parser.add_argument('--screen-margin', type=float, default=.05, help='屏幕设备检测框每侧外扩比例')
    parser.add_argument('--margin', type=float, default=.2, help='人脸检测框每侧外扩比例')
    parser.add_argument('--hold', type=float, default=.3, help='检测前后延长遮罩秒数，缓解漏检和闪烁')
    parser.add_argument('--block-size', type=int, default=32, help='原图马赛克块尺寸（像素）')
    parser.add_argument('--encoder', choices=['auto', 'cpu', 'nvenc'], default='auto',
                        help='auto 实测 NVENC，不可用时回退 CPU')
    parser.add_argument('--render-mode', choices=['auto', 'full', 'sparse'], default='auto',
                        help='auto=兼容时仅重编码需打码的关键帧组；full=完整编码；sparse=强制分段编码')
    parser.add_argument('--preset', default='superfast',
                        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow'],
                        help='CPU 编码预设')
    parser.add_argument('--quality', type=int, default=23, help='CRF/CQ，越小画质越高、文件越大')
    parser.add_argument('--jobs', type=int, default=1, help='同时处理视频数；CPU 建议 2–4')
    parser.add_argument('--threads', type=int, default=4, help='每个视频的解码/滤镜/编码线程数')
    parser.add_argument('--start', type=float, default=0, help='从指定秒数开始（预览）')
    parser.add_argument('--duration', type=float, help='只处理指定秒数（预览）')
    parser.add_argument('--scan-only', action='store_true', help='只检测并缓存，不输出视频')
    parser.add_argument('--rescan', action='store_true', help='忽略检测缓存')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已有输出，仍可复用检测缓存')
    parser.add_argument('--model', type=Path, default=MODEL, help='YuNet 2023mar ONNX 模型路径')
    args = parser.parse_args(argv)
    if args.detect_fps < 0 or args.detect_size < 64 or not 0 < args.threshold < 1:
        parser.error('detect-fps 必须 >=0，detect-size 必须 >=64，threshold 必须在 (0,1)')
    if not 0 < args.screen_threshold < 1 or args.screen_margin < 0:
        parser.error('screen-threshold 必须在 (0,1)，screen-margin 必须非负')
    if min(args.jobs, args.threads, args.block_size) < 1 or args.margin < 0 or args.hold < 0:
        parser.error('jobs/threads/block-size 必须为正，margin/hold 必须非负')
    if args.start < 0 or (args.duration is not None and args.duration <= 0) or not 0 <= args.quality <= 51:
        parser.error('start 必须非负，duration 必须为正，quality 必须在 0–51')
    if args.scan_only and args.mode == 'mute':
        parser.error('--scan-only 不适用于 --mode mute')
    args.targets = sorted(set(args.targets))
    for key in ('input', 'output', 'model', 'screen_model'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    return args


def main(argv=None):
    args = parse_args(argv)
    for executable in ('ffmpeg', 'ffprobe'):
        if not shutil.which(executable):
            raise ValueError(f'请先安装 {executable}')
    if not args.input.exists():
        raise ValueError(f'输入不存在：{args.input}')
    if args.input.is_dir() and (args.output == args.input or args.output.is_relative_to(args.input)):
        raise ValueError('输出目录必须位于输入目录之外，以免重复处理输出')
    sources = [args.input] if args.input.is_file() else sorted(
        p for p in args.input.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not sources:
        raise ValueError('未找到支持的视频文件')
    tasks = []
    seen = set()
    for path in sources:
        relative = Path(path.name) if args.input.is_file() else path.relative_to(args.input)
        target = args.output / relative.with_suffix('.mp4')
        if target == path or target in seen:
            raise ValueError(f'输出路径冲突：{target}；请调整输出目录或视频名称')
        seen.add(target)
        tasks.append((path, target))
    if args.mode != 'mute':
        import cv2
        # OpenCV's global pool is shared by batch workers; avoid CPU oversubscription.
        cv2.setNumThreads(args.threads)
        if 'face' in args.targets:
            ensure_model(args.model)
        if 'screen' in args.targets:
            screen_detector.ensure_model(args.screen_model)
    encoder = choose_encoder(args.encoder) if args.mode != 'mute' and not args.scan_only else 'libx264'
    log(f'共 {len(tasks)} 个视频，并发 {args.jobs}，输出：{args.output}')
    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(process_video, path, output, args, encoder): path for path, output in tasks}
        try:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    failures.append({'input': str(futures[future]), 'error': str(exc)})
                    log(f'[失败] {futures[future].name}：{exc}')
        except KeyboardInterrupt:
            # Let workers receive a shared stop request at their next frame/progress line.
            STOP.set()
            for future in futures:
                future.cancel()
            log('正在保存检测进度并停止…')
            raise
    if failures:
        atomic_json(args.output / 'failures.json', failures)
        return 1
    (args.output / 'failures.json').unlink(missing_ok=True)
    return 0


import threading
STOP = threading.Event()

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as error:
        log(f'错误：{error}')
        sys.exit(1)
