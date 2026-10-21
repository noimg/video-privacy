"""H.264 sparse masking: encode affected IDR groups, remux all other packets.

The narrow compatibility checks are deliberate. Other formats use the existing
FFmpeg renderer. avc3 stores changing SPS/PPS in-band at splice boundaries.
"""
from __future__ import annotations

from bisect import bisect_right
import math
from pathlib import Path
import time


class SparseUnsupported(ValueError):
    """Input cannot be spliced safely; the caller may use full transcoding."""


def active_intervals(changes, duration):
    intervals = []
    start = None
    for t, boxes in changes:
        if boxes and start is None:
            start = t
        elif not boxes and start is not None:
            intervals.append((start, min(t, duration)))
            start = None
    if start is not None:
        intervals.append((start, duration))
    return intervals


def contains_idr(packet):
    """Check length-prefixed AVC packets, including nondefault NAL length sizes."""
    extra = packet.stream.codec_context.extradata or b''
    if len(extra) < 5 or extra[0] != 1:
        return False
    length_size = (extra[4] & 3) + 1
    data = memoryview(packet)
    offset = 0
    while offset + length_size < len(data):
        size = int.from_bytes(data[offset:offset + length_size], 'big')
        offset += length_size
        if size <= 0 or offset + size > len(data):
            return False
        if data[offset] & 31 == 5:
            return True
        offset += size
    return False


def mask_frame(frame, boxes, style, block_size):
    """Copy decoder references before modifying YUV planes; never convert to RGB."""
    import av
    import cv2
    import numpy as np

    if not boxes:
        return frame
    if frame.format.name != 'yuv420p':
        raise SparseUnsupported(f'解码像素格式 {frame.format.name} 不支持分段编码')
    result = av.VideoFrame(frame.width, frame.height, 'yuv420p')
    # Chroma and color metadata must survive both masking and splice boundaries.
    result.pts, result.time_base = frame.pts, frame.time_base
    for name in ('color_range', 'colorspace', 'color_primaries', 'color_trc'):
        setattr(result, name, getattr(frame, name))
    for i, (source, dest) in enumerate(zip(frame.planes, result.planes)):
        src = np.frombuffer(source, dtype=np.uint8).reshape(source.height, source.line_size)
        dst = np.frombuffer(dest, dtype=np.uint8).reshape(dest.height, dest.line_size)
        dst[:, :source.width] = src[:, :source.width]
        factor = 1 if i == 0 else 2
        for x, y, w, h in boxes:
            if style == 'mosaic':
                # Expand to mosaic cells so even small boundary faces are fully covered.
                x0 = max(0, math.floor(x / block_size) * block_size)
                y0 = max(0, math.floor(y / block_size) * block_size)
                x1 = min(frame.width, math.ceil((x + w) / block_size) * block_size)
                y1 = min(frame.height, math.ceil((y + h) / block_size) * block_size)
            else:
                x0, y0, x1, y1 = max(0, x), max(0, y), min(frame.width, x + w), min(frame.height, y + h)
            x0, y0, x1, y1 = x0 // factor, y0 // factor, math.ceil(x1 / factor), math.ceil(y1 / factor)
            if x1 <= x0 or y1 <= y0:
                continue
            roi = dst[y0:y1, x0:x1]
            if style == 'black':
                roi[:] = (0 if frame.color_range == 2 else 16) if i == 0 else 128
            else:
                small = cv2.resize(roi, (max(1, math.ceil((x1 - x0) * factor / block_size)),
                                         max(1, math.ceil((y1 - y0) * factor / block_size))),
                                   interpolation=cv2.INTER_AREA)
                roi[:] = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    return result


def render_sparse(path, output, info, args, changes, stop, log):
    import av

    if info['codec'] != 'h264' or info['rotation'] or info['width'] % 2 or info['height'] % 2:
        raise SparseUnsupported('分段编码仅支持无旋转、偶数尺寸的 H.264 视频')
    if args.start:
        raise SparseUnsupported('非零 --start 使用完整编码以保证截取准确')
    if args.quality == 0:
        raise SparseUnsupported('无损编码的 H.264 profile 切换需要完整编码以保证播放器兼容')
    duration = min(info['duration'], args.duration or math.inf)
    intervals = active_intervals(changes, duration)
    interval_ends = [b for _, b in intervals]
    change_times = [t for t, _ in changes]
    began = last_log = time.monotonic()
    stats = {'strategy': 'sparse', 'encoder': 'libx264 + stream-copy',
             'copied_frames': 0, 'encoded_frames': 0, 'copied_seconds': 0.0,
             'encoded_seconds': 0.0, 'encoded_gops': 0, 'copied_gops': 0}
    with av.open(str(path)) as source:
        stream = next(s for s in source.streams.video if s.index == info['video_index'])
        ctx = stream.codec_context
        if (ctx.has_b_frames or ctx.format is None or ctx.format.name != 'yuv420p'
                or stream.start_time not in (None, 0)):
            raise SparseUnsupported('含 B 帧、非 YUV420P 或非零视频起点时使用完整编码')
        if not ctx.extradata or ctx.extradata[0] != 1:
            raise SparseUnsupported('分段编码需要 AVC 格式的 H.264 参数集')
        audio = list(source.streams.audio) if args.mode == 'blur' else []
        if any(s.codec_context.name not in ('aac', 'mp3', 'alac', 'ac3', 'eac3') for s in audio):
            raise SparseUnsupported('此音频编码不能直接封装为 MP4')
        ctx.thread_count = args.threads
        ctx.thread_type = 'AUTO'
        tb = stream.time_base
        rate = stream.guessed_rate or stream.average_rate
        if not rate or rate <= 0:
            raise SparseUnsupported('无法确定视频帧率')
        stream_index = stream.index
        with av.open(str(output), 'w', format='mp4', options={'movflags': '+faststart'}) as target:
            out = target.add_stream_from_template(stream)
            # Every original IDR restores original SPS/PPS via the bitstream filter;
            # every encoded IDR includes its encoder's SPS/PPS. avc3 signals this.
            bsf = av.BitStreamFilterContext('h264_mp4toannexb', stream, out)
            out.codec_context.codec_tag = 'avc3'
            out.time_base = tb
            audio_map = {s.index: target.add_stream_from_template(s) for s in audio}
            previous_pts = None
            last_end = 0
            group = []
            group_bytes = 0

            def emit(packet):
                packet.stream = out
                target.mux(packet)

            def encode_group(packets):
                encoder = av.CodecContext.create('libx264', 'w')
                encoder.width, encoder.height = ctx.width, ctx.height
                encoder.pix_fmt = 'yuv420p'
                encoder.time_base, encoder.framerate = tb, rate
                encoder.thread_count = args.threads
                encoder.sample_aspect_ratio = ctx.sample_aspect_ratio or 1
                for name in ('color_range', 'colorspace', 'color_primaries', 'color_trc'):
                    setattr(encoder, name, getattr(ctx, name))
                encoder.options = {'preset': args.preset, 'crf': str(args.quality),
                                   'tune': 'zerolatency', 'bf': '0',
                                   'profile': 'baseline',
                                   'x264-params': 'repeat-headers=1:annexb=1:scenecut=0:open-gop=0'}
                encoder.open()
                durations = {p.pts: p.duration for p in packets}
                expected = set(durations)
                actual = set()
                ctx.flush_buffers()

                def encode_frame(frame):
                    frame.time_base = tb
                    if frame.width != info['width'] or frame.height != info['height']:
                        raise SparseUnsupported('不支持视频中途改变分辨率')
                    t = float(frame.pts * frame.time_base)
                    index = bisect_right(change_times, t) - 1
                    boxes = changes[index][1] if index >= 0 else []
                    masked = mask_frame(frame, boxes, args.style, args.block_size)
                    # Avoid carrying the original I/P picture type to the new encoder.
                    masked.pict_type = 0
                    return encoder.encode(masked)

                def write_encoded(packet):
                    if packet.pts not in expected or packet.dts != packet.pts or packet.pts in actual:
                        raise SparseUnsupported('分段编码后帧时间戳发生变化')
                    packet.duration = durations[packet.pts]
                    actual.add(packet.pts)
                    emit(packet)

                for packet in packets:
                    if stop.is_set():
                        raise InterruptedError('已停止分段编码')
                    for frame in ctx.decode(packet):
                        for encoded in encode_frame(frame):
                            write_encoded(encoded)
                for frame in ctx.decode(None):
                    for encoded in encode_frame(frame):
                        write_encoded(encoded)
                for encoded in encoder.encode(None):
                    write_encoded(encoded)
                if actual != expected:
                    raise SparseUnsupported('分段编码后帧数发生变化')
                stats['encoded_frames'] += len(actual)

            def process_group(packets, end):
                nonlocal last_log
                begin = float(packets[0].pts * tb)
                end = float(end * tb)
                idx = bisect_right(interval_ends, begin)
                affected = idx < len(intervals) and intervals[idx][0] < end
                if affected:
                    encode_group(packets)
                    stats['encoded_gops'] += 1
                    stats['encoded_seconds'] += end - begin
                else:
                    for packet in packets:
                        for filtered in bsf.filter(packet):
                            emit(filtered)
                    stats['copied_frames'] += len(packets)
                    stats['copied_gops'] += 1
                    stats['copied_seconds'] += end - begin
                now = time.monotonic()
                if now - last_log >= 15:
                    log(f'[{Path(path).name}] 分段输出 {end:.0f}/{duration:.0f}s，'
                        f'{end / max(.001, now - began):.1f} 倍速，'
                        f'已重编码 {stats["encoded_seconds"]:.0f}s')
                    last_log = now

            log(f'[{Path(path).name}] 稀疏打码：只编码含遮罩的关键帧组，其余复制视频流')
            for packet in source.demux([stream, *audio]):
                if stop.is_set():
                    raise InterruptedError('已停止分段编码')
                if packet.dts is None:
                    continue
                if packet.stream.index != stream_index:
                    # Keep source audio timing and pre-roll intact for full videos.
                    if args.duration and packet.pts is not None and float(packet.pts * packet.time_base) >= duration:
                        continue
                    packet.stream = audio_map[packet.stream.index]
                    target.mux(packet)
                    continue
                if packet.pts is None or packet.dts != packet.pts:
                    raise SparseUnsupported('分段编码要求视频 PTS 与 DTS 一致')
                if previous_pts is not None and packet.pts <= previous_pts:
                    raise SparseUnsupported('视频时间戳不递增')
                if float(packet.pts * tb) >= duration:
                    break
                previous_pts = packet.pts
                if packet.is_keyframe:
                    if not contains_idr(packet):
                        raise SparseUnsupported('关键帧不是独立 IDR，不能安全拼接')
                    if group:
                        process_group(group, packet.pts)
                        group, group_bytes = [], 0
                if not group and not packet.is_keyframe:
                    raise SparseUnsupported('视频没有从独立关键帧开始')
                if packet.duration <= 0:
                    raise SparseUnsupported('视频包缺少帧时长')
                group.append(packet)
                group_bytes += packet.size
                last_end = packet.pts + packet.duration
                if group_bytes > 128 * 1024 * 1024 or len(group) > 3000:
                    raise SparseUnsupported('关键帧间隔过长，使用完整编码以限制内存')
            if group:
                process_group(group, last_end)
            if not stats['copied_frames'] + stats['encoded_frames']:
                raise SparseUnsupported('没有可处理的视频帧')
            if float(last_end * tb) < duration - max(1.0, 3 / max(info['fps'], 1)):
                raise RuntimeError('源视频提前结束，拒绝发布不完整输出')
    stats['seconds'] = time.monotonic() - began
    total = stats['copied_frames'] + stats['encoded_frames']
    log(f'[{Path(path).name}] 复制 {stats["copied_frames"]}/{total} 帧 '
        f'({stats["copied_frames"] / total:.1%})，重编码 {stats["encoded_seconds"]:.1f}s')
    return stats
