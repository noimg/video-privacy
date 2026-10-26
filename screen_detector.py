"""Local COCO device detection for privacy masks (YOLOX-S, Apache-2.0).

Masks deliberately cover the full device box: cropping a laptop to an assumed
screen fraction can expose tilted screens. No images leave this process.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import threading
import urllib.request

MODEL = Path(__file__).resolve().parent / 'models/object_detection_yolox_2022nov.onnx'
MODEL_SHA256 = 'c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'
MODEL_URL = ('https://media.githubusercontent.com/media/opencv/opencv_zoo/main/'
             'models/object_detection_yolox/object_detection_yolox_2022nov.onnx')
CLASSES = {62: 'monitor', 63: 'laptop', 67: 'phone'}
VERSION = 1
_MODEL_LOCK = threading.Lock()


def ensure_model(path):
    """Download once, validate before publishing, and never accept a partial file."""
    path = Path(path)
    with _MODEL_LOCK:
        if not path.exists():
            if path.resolve() != MODEL:
                raise ValueError(f'找不到屏幕检测模型：{path}')
            path.parent.mkdir(parents=True, exist_ok=True)
            print('下载 YOLOX-S 屏幕检测模型（约 36 MB）…', flush=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
                temporary = Path(tmp.name)
            try:
                with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
                    with temporary.open('wb') as output:
                        shutil.copyfileobj(response, output)
                if hashlib.sha256(temporary.read_bytes()).hexdigest() != MODEL_SHA256:
                    raise ValueError('屏幕检测模型校验失败')
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.resolve() == MODEL and sha != MODEL_SHA256:
            raise ValueError('屏幕检测模型校验失败，请删除后重新下载')
        return sha


class ScreenDetector:
    def __init__(self, model, threshold=.3, device='auto', threads=4):
        import numpy as np
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError('屏幕检测需要 ONNX Runtime，请运行 pip install -r requirements.txt') from exc

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        available = ort.get_available_providers()
        cuda = device != 'cpu' and 'CUDAExecutionProvider' in available
        if device == 'cuda' and not cuda:
            raise ValueError('CUDA 屏幕检测不可用，请安装 onnxruntime-gpu 及其 CUDA/cuDNN 依赖，或使用 --screen-device cpu')
        if cuda and hasattr(ort, 'preload_dlls'):
            # Also find CUDA libraries installed as NVIDIA Python packages.
            ort.preload_dlls()
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=providers)
        self.backend = 'cuda' if 'CUDAExecutionProvider' in self.session.get_providers() else 'cpu'
        if device == 'cuda' and self.backend != 'cuda':
            raise ValueError('CUDA 屏幕检测初始化失败，请检查 CUDA/cuDNN 依赖')
        self.input_name = self.session.get_inputs()[0].name
        self.threshold = threshold
        grids, strides = [], []
        for stride in (8, 16, 32):
            y, x = np.mgrid[:640 // stride, :640 // stride]
            grids.append(np.stack((x, y), axis=-1).reshape(-1, 2))
            strides.append(np.full(((640 // stride) ** 2, 1), stride))
        self.grids = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)

    def detect(self, image):
        import cv2
        import numpy as np

        h, w = image.shape[:2]
        scale = min(640 / w, 640 / h)
        rw, rh = max(1, int(w * scale)), max(1, int(h * scale))
        padded = np.full((640, 640, 3), 114, dtype=np.uint8)
        padded[:rh, :rw] = cv2.resize(image, (rw, rh))
        # The zoo model expects RGB float32 in [0,255], with top-left letterboxing.
        blob = cv2.dnn.blobFromImage(padded, swapRB=True)
        output = self.session.run(None, {self.input_name: blob})[0]
        return self.postprocess(output, w, h, scale)

    def postprocess(self, output, width, height, scale):
        import cv2
        import numpy as np

        if output.shape != (1, 8400, 85):
            raise ValueError(f'屏幕模型输出不兼容：{output.shape}，需要 YOLOX-S 640×640 COCO ONNX')
        dets = output[0]
        # Choose the best COCO class before filtering devices, avoiding e.g. a
        # confident book being masked just because its phone score is nonzero.
        labels = np.argmax(dets[:, 5:], axis=1)
        scores = dets[:, 4] * dets[np.arange(len(dets)), labels + 5]
        selected = np.flatnonzero(np.isin(labels, list(CLASSES)) & (scores >= self.threshold))
        if not len(selected):
            return []
        boxes = dets[selected, :4].copy()
        boxes[:, :2] = (boxes[:, :2] + self.grids[selected]) * self.strides[selected]
        boxes[:, 2:4] = np.exp(boxes[:, 2:4]) * self.strides[selected]
        boxes[:, :2] -= boxes[:, 2:4] / 2
        boxes /= scale
        # Remove padding-only predictions, and clamp all boxes to actual pixels.
        ends = np.minimum(boxes[:, :2] + boxes[:, 2:4], [width, height])
        boxes[:, :2] = np.maximum(boxes[:, :2], 0)
        boxes[:, 2:4] = ends - boxes[:, :2]
        finite = np.isfinite(boxes).all(axis=1) & (boxes[:, 2:4] > 0).all(axis=1)
        result = []
        # NMS per device class: a phone lying over a laptop must survive.
        for class_id, name in CLASSES.items():
            indices = np.flatnonzero(finite & (labels[selected] == class_id))
            if not len(indices):
                continue
            keep = cv2.dnn.NMSBoxes(boxes[indices].tolist(), scores[selected[indices]].tolist(),
                                    self.threshold, .45)
            for i in np.asarray(keep).reshape(-1):
                j = indices[i]
                x, y, bw, bh = boxes[j]
                result.append([float(x / width), float(y / height), float(bw / width),
                               float(bh / height), float(scores[selected[j]]), name])
        return result
