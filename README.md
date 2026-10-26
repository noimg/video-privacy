# 视频静音与人脸打码

批量处理长视频，支持去音频、人脸马赛克或黑块、多人脸检测、并发和检测缓存。源文件保持不变。

## 安装

需要 Python 3.10+ 和 FFmpeg 4.4+（包含 libx264）。

```bash
sudo apt-get install ffmpeg
python -m pip install -r requirements.txt
```

已附带轻量 YuNet 人脸检测模型，无需额外下载。模型来源和许可证见 [models/README.md](models/README.md)。

## 使用

```bash
# 静音 + 人脸马赛克；输入可以是文件或目录
python video_privacy.py ./Videos-origin -o ./Videos-processed --jobs 2

# 仅去音频，直接复制视频流，不损失视频画质
python video_privacy.py ./Videos-origin -o ./Videos-muted --mode mute

# 仅人脸打码，保留音频
python video_privacy.py ./Videos-origin -o ./Videos-blurred --mode blur

# 使用黑块遮挡
python video_privacy.py ./Videos-origin -o ./Videos-black --style black

# 先处理前两分钟，检查效果
python video_privacy.py ./Videos-origin -o ./Videos-preview --duration 120
```

目录会递归处理，输出统一为 MP4，并保留子目录结构。不指定路径时，默认读取项目同级的 `Videos-origin`，输出到同级的 `Videos-processed`。

## 检测与速度

默认每秒检测 5 帧，检测图最长边为 640 像素；遮罩扩大并前后延长，减少运动造成的漏遮。抽帧可能漏掉短暂出现的人脸，小脸、侧脸和遮挡也可能漏检。需要更高覆盖率时，可逐帧检测并提高分辨率：

```bash
python video_privacy.py ./Videos-origin -o ./Videos-careful --detect-fps 0 --detect-size 1280 --threshold 0.5
```

CPU 默认采用分段编码：**只重编码需要打码的关键帧组，其余直接复制视频流**，适合人脸出现频率低的长视频。检测频率不变，关键帧组内也只遮挡原本需要打码的时刻。直接复制的画面保留原始画质。

```bash
# 复用已有检测缓存，重新生成输出（例如调整遮盖范围后）
python video_privacy.py ./Videos-origin -o ./Videos-processed --encoder cpu --overwrite

# 使用完整编码，提高播放器兼容性
python video_privacy.py ./Videos-origin -o ./Videos-full --render-mode full
```

分段编码支持无 B 帧、无旋转的偶数尺寸 H.264 / YUV420P 视频。其他格式、非零 `--start`、无损编码（`--quality 0`）会自动回退完整编码；`--render-mode sparse` 可强制分段处理并在不兼容时报告错误。输出使用 MP4 / avc3 封装，若播放器兼容性有问题可改用 `--render-mode full`。

工具自动试用 NVIDIA NVENC，不可用时回退 CPU；使用 NVENC 时仍采用完整编码。批量处理长视频时，建议从 `--jobs 2` 开始调整并发。新视频需先扫描全片，再生成输出；实际耗时取决于视频长度、人脸出现频率和硬件性能。

## 缓存与参数

检测进度每 30 秒保存一次，Ctrl+C 后重新运行相同命令可继续扫描；编码中断后需重新编码当前视频。已完成的输出会跳过，使用 `--overwrite` 可重新生成。

常用参数：`--jobs` 并发数、`--threads` 每线程池线程数、`--block-size` 马赛克块大小、`--margin` 遮罩外扩比例（默认每侧 20%，可调小以缩小遮盖范围）、`--hold` 遮罩前后延长秒数。调整打码样式可复用检测缓存，修改检测参数会自动重扫。

例如，将每侧外扩比例缩小到 10%，并重新生成已有输出：

```bash
python video_privacy.py ./Videos-origin -o ./Videos-processed --margin 0.1 --overwrite
```

建议先用 `--duration 120` 预览效果，确认遮盖范围后再处理全片。查看全部参数：

```bash
python video_privacy.py --help
```
