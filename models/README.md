# YuNet 人脸检测模型

- 来源：https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- 文件：`face_detection_yunet_2023mar.onnx`，OpenCV FaceDetectorYN 接口。
- SHA-256：`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
- 上游模型目录采用 MIT 许可证；许可证原文见 `LICENSE-YuNet`。

检测在本机完成，不上传视频或图像。

## YOLOX-S 手机/电脑设备检测模型

- 来源：https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox
- 文件：`object_detection_yolox_2022nov.onnx`，约 36 MB，首次启用屏幕检测时自动下载；不纳入 Git。
- 下载：https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/object_detection_yolox/object_detection_yolox_2022nov.onnx
- SHA-256：`c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`
- Apache 2.0 许可证，原文见 [LICENSE-YOLOX](LICENSE-YOLOX)。上游作者：Megvii；OpenCV Zoo 提供 ONNX 模型。
- 使用 COCO 类别 `cell phone`（67）、`laptop`（63）、`tv`（62，包含显示器），遮住设备整体边界框。推理由 ONNX Runtime 执行，支持 CPU / CUDA。
- `--screen-model` 可指定相同输入/输出格式的本地 YOLOX-S ONNX；其他 YOLO 系列模型格式不兼容。

无需 Hugging Face。如需手动下载，将上述文件放到此目录即可，工具会校验内置路径的模型哈希；网络失败不会留下被误用的半成品模型。
