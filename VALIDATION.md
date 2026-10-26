# 屏幕打码验证（2026-10-26）

素材：`../videos/20260906-Yangyang-1.mp4`，5687.7 秒（约 95 分钟），1600×1200、约 30 fps、H.264 / YUV420P、无 B 帧。环境为 RTX 4090、ONNX Runtime GPU 1.31、OpenCV 4.14、PyAV 18；CUDA 屏幕推理可用，NVENC 实测不可用，因此使用 CPU 编码。

## 实测结果

| 原片范围 | 检测对象 / 频率 | 扫描 | 输出 | 总耗时 | 编码方式 |
| --- | --- | ---: | ---: | ---: | --- |
| 0–300 秒 | 屏幕 / 2 fps | 76.3 秒 | 13.9 秒 | 91.7 秒 | 分段编码，94% 帧直接复制 |
| 0–15 秒 | 人脸和屏幕 / 5 fps | 9.3 秒 | 11.3 秒 | 22.2 秒 | 分段编码，全部 GOP 含遮罩 |
| 2690–2750 秒 | 人脸和屏幕 / 5 fps | 32.2 秒 | 88.1 秒 | 122.0 秒 | 非零 start，完整编码 |

第一项采用 4 线程、黑块，600 个采样帧中有 13 帧检测到手机。8998 个输出帧中复制 8458 帧、重编码 540 帧（18.0 秒），总处理速度约 3.3 倍。输出完整解码无错误；第 60 秒和第 200 秒未遮挡帧的时间戳及 YUV 像素与源视频一致。

手机预览采用 4 线程、马赛克，已查看输出画面，手机和同时出现的人脸均有遮罩。电脑预览采用 2 线程、黑块，300 个采样帧均检测到设备，包含笔记本和背景显示器；已查看输出画面，输出分辨率、时长与静音均验证通过。统计中的设备数按每个采样帧累加，不是去重后的实体数量。

本次为片段验证。屏幕持续出现时需重编码整个相应时段，速度会明显低于稀疏出现的片段；全片耗时取决于目标分布和硬件。2 fps 比默认 5 fps 更快，也更容易漏掉短暂画面。未进行全片输出或人工标注的漏检率评估。

## 复现命令

```bash
python video_privacy.py ../videos -o ../videos-screen-benchmark --targets screen --detect-fps 2 --duration 300 --threads 4 --encoder cpu --style black
python video_privacy.py ../videos -o ../videos-phone-preview --targets face screen --duration 15 --threads 4 --encoder cpu
python video_privacy.py ../videos -o ../videos-screen-preview --targets face screen --start 2690 --duration 60 --threads 2 --style black --encoder auto
```

对应目录包含 MP4 与 `.report.json`；`../videos-screen-preview/comparison.jpg` 包含手机和电脑画面的处理前后对比。重跑时可加 `--overwrite` 复用检测缓存并重新编码，测速时加 `--rescan --overwrite` 才会重新扫描。

## 自动测试

`python -m unittest discover -s tests -v`：28 项全部通过。覆盖三类设备的边界框与补边坐标、同类去重、重叠设备保留、RGB 输入、CUDA 不可用时报错、屏幕单独检测、联合检测、缓存复用与失效、断点续扫、遮罩边距、静音绕过模型，以及原有的音频、旋转、VFR 和分段编码测试。自动测试不需要下载屏幕模型或使用 GPU；屏幕模型推理另外通过上述实片验证。
