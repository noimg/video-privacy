# 视频静音、人脸与屏幕打码

批量处理长视频，支持去音频、人脸和手机/电脑屏幕马赛克或黑块、多目标检测、并发和检测缓存。源文件保持不变。

## 安装

需要 Python 3.10+ 和 FFmpeg 4.4+（包含 libx264）。

```bash
sudo apt-get install ffmpeg
python -m pip install -r requirements.txt
```

已附带轻量 YuNet 人脸检测模型。首次启用屏幕检测时自动下载约 36 MB 的 YOLOX-S 模型并校验 SHA-256，之后离线运行。模型来源和许可证见 [models/README.md](models/README.md)。视频与图像均在本机处理。

屏幕检测默认使用 ONNX Runtime CPU；有 NVIDIA GPU 时可替换为 GPU 版本，加速长视频扫描（运行时还需对应的 CUDA/cuDNN，下面的 extras 会安装 NVIDIA Python 库）：

```bash
python -m pip uninstall -y onnxruntime
python -m pip install 'onnxruntime-gpu[cuda,cudnn]>=1.21,<2'
```

CPU 版和 GPU 版选择一种安装。`--screen-device auto` 优先使用可用的 CUDA，启动日志会显示实际设备；`--screen-device cuda` 在 CUDA 无法初始化时直接报错。

## 使用

```bash
# 静音 + 人脸、手机、笔记本和显示器打码
python video_privacy.py ../Videos-origin -o ../Videos-processed --targets face screen

# 仅给手机和电脑屏幕打码，保留音频
python video_privacy.py ../Videos-origin -o ../Videos-processed --targets screen --mode blur

# 静音 + 人脸马赛克；输入可以是文件或目录
python video_privacy.py ../Videos-origin -o ../Videos-processed --jobs 2

# 仅去音频，直接复制视频流，不损失视频画质
python video_privacy.py ../Videos-origin -o ../Videos-processed --mode mute

# 仅人脸打码，保留音频
python video_privacy.py ../Videos-origin -o ../Videos-processed --mode blur

# 使用黑块遮挡
python video_privacy.py ../Videos-origin -o ../Videos-processed --style black

# 先处理前两分钟，检查效果
python video_privacy.py ../Videos-origin -o ../Videos-processed --duration 120
```

`--targets` 默认仍为 `face`；启用新功能时指定 `screen`，或指定 `face screen` 同时处理。屏幕检测识别手机、笔记本和显示器/电视，**遮罩覆盖整台设备的矩形框**，包括笔记本键盘、设备边框以及框内背景；不按固定比例裁切屏幕，以覆盖倾斜、竖屏和部分遮挡的设备。默认每侧外扩 5%，用 `--screen-margin` 调整；人脸继续使用 `--margin`（默认 20%）。

细小、模糊、被遮挡或占满画面而缺少设备外观的屏幕可能漏检。屏幕检测置信度默认为 `0.3`，可用 `--screen-threshold 0.2` 增加召回率，同时也会增加误检。需要遮住细小文字时，建议 `--style black`，或增大 `--block-size` 后检查马赛克效果。

目录会递归处理，输出统一为 MP4，并保留子目录结构。不指定路径时，默认输入目录为 `../Videos-origin`，默认输出目录为 `../Videos-processed`（均位于项目同级）。

## 检测与速度

默认每秒检测 5 帧，人脸检测图最长边为 640 像素，屏幕模型固定使用 640×640 输入并等比补边；两类检测共用一次解码和取样。解码/缩放与检测流水执行；同时启用人脸和 CUDA 屏幕检测时，两种模型并行推理。预取最多一张待检测图，避免长视频占用不断增长的内存；原有固定频率采样和检测缓存保持兼容。遮罩扩大并前后延长，减少运动造成的漏遮。抽帧可能漏掉短暂出现的目标，小脸、侧脸和遮挡也可能漏检。需要更高覆盖率时，可逐帧检测并提高人脸检测分辨率：

```bash
python video_privacy.py ../Videos-origin -o ../Videos-processed --detect-fps 0 --detect-size 1280 --threshold 0.5
```

后续视频中人脸和屏幕较少时，可选择自适应扫描：

```bash
python video_privacy.py ../Videos-origin -o ../Videos-processed --targets face screen --scan-mode adaptive --encoder cpu
```

`--scan-mode adaptive` 先以 `--coarse-fps 1` 初筛全片，再以 `--detect-fps 5` 补扫命中位置前后各 `--refine-window 2` 秒。相邻片段会合并，补扫复用初筛帧的结果，并通过关键帧定位跳过片段间的空档。两个阶段都支持断点续扫。**初筛间隔内短暂出现且未被命中的目标可能完全漏掉**，补扫无法找回所有这样的目标；实测一个 60 秒片段中，固定 5 fps 有 31 组检测，自适应有 28 组。因此默认仍为 `--scan-mode uniform`。需要减少这个风险时可提高 `--coarse-fps 2`，或继续使用默认模式；目标持续出现时，自适应的第二次解码可能使它比固定频率扫描稍慢。

在 RTX 4090 上，三段原视频实测的默认扫描较优化前提速约 2.3–2.6 倍，采样数和检测结果一致；60 秒无目标片段由 19.8 秒降至默认模式的 7.5 秒，自适应模式为 5.5 秒。这些是检测阶段的片段测量，不是整片处理时间；完整数据见 [VALIDATION.md](VALIDATION.md)。

CPU 编码默认采用分段编码：**只重编码需要打码的关键帧组，其余直接复制视频流**，人脸和屏幕均适用。适合目标出现频率低的长视频。检测频率不变，关键帧组内也只遮挡原本需要打码的时刻。直接复制的画面保留原始画质。检测用 GPU 和编码用 CPU 可独立选择。

```bash
# 长视频：GPU 检测 + 仅重编码需打码片段
python video_privacy.py ../Videos-origin -o ../Videos-processed --targets face screen --screen-device cuda --encoder cpu

# 先检查含电脑的片段（该预览从第 2690 秒开始）
python video_privacy.py ../Videos-origin -o ../Videos-processed --targets face screen --start 2690 --duration 60 --style black

# 优先处理速度：每秒检测 2 帧；会增加短暂目标漏检风险
python video_privacy.py ../Videos-origin -o ../Videos-processed --targets face screen --detect-fps 2 --encoder cpu

# 复用已有检测缓存，重新生成输出（例如调整遮盖范围后）
python video_privacy.py ../Videos-origin -o ../Videos-processed --encoder cpu --overwrite

# 使用完整编码，提高播放器兼容性
python video_privacy.py ../Videos-origin -o ../Videos-processed --render-mode full
```

分段编码支持无 B 帧、无旋转的偶数尺寸 H.264 / YUV420P 视频。其他格式、非零 `--start`、无损编码（`--quality 0`）会自动回退完整编码；`--render-mode sparse` 可强制分段处理并在不兼容时报告错误。输出使用 MP4 / avc3 封装，若播放器兼容性有问题可改用 `--render-mode full`。

工具自动试用 NVIDIA NVENC，不可用时回退 CPU；使用 NVENC 时仍采用完整编码。批量处理长视频时，建议从 `--jobs 2` 开始调整并发；GPU 内存不足时降低并发。新视频需先扫描全片，再生成输出；实际耗时取决于视频长度、目标出现频率和硬件性能。屏幕持续出现时，相应时段需全部重编码。

## 缓存与参数

检测进度每 30 秒保存一次，Ctrl+C 后重新运行相同命令可继续扫描；编码中断后需重新编码当前视频。已完成的输出会跳过，使用 `--overwrite` 可重新生成。

常用参数：`--jobs` 并发数、`--threads` 每线程池线程数、`--block-size` 马赛克块大小、`--margin` 人脸遮罩外扩比例、`--screen-margin` 设备遮罩外扩比例、`--hold` 遮罩前后延长秒数。调整样式或遮罩边距可复用检测缓存，修改检测对象、模型、频率或阈值会自动重扫。原有人脸缓存仍可复用；屏幕和联合检测使用独立缓存，避免把未检测屏幕的旧结果当成“无屏幕”。自适应与固定频率扫描也使用独立缓存；调整自适应初筛频率或补扫窗口会使其缓存失效。`--scan-only` 可仅扫描不编码，输出报告包含人脸/屏幕采样帧数、手机/笔记本/显示器检测数、实际推理设备与处理耗时；自适应输出另有 `coarse_samples`、`refine_samples` 和 `refine_seconds`，分别记录初筛推理帧数、额外补扫推理帧数和补扫片段总长度。

例如，将每侧外扩比例缩小到 10%，并重新生成已有输出：

```bash
python video_privacy.py ../Videos-origin -o ../Videos-processed --margin 0.1 --overwrite
```

建议先用 `--duration 120` 预览效果，确认遮盖范围后再处理全片。查看全部参数：

```bash
python video_privacy.py --help
```

## 验证

```bash
python -m unittest discover -s tests -v
```

已通过 36 项测试，并使用原视频的手机、笔记本和显示器片段进行实测。早期 5 分钟屏幕打码测试耗时 91.7 秒，直接复制 94% 的视频帧；本次扫描加速的对比、参数、环境与性能限制见 [VALIDATION.md](VALIDATION.md)。
