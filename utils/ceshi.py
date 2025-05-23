from mobilenet_feat import predict_keyframes_with_onnx

# ✅ 输入参数
frame_dir = "../outputs/frames/sanxingdui1"                # 替换为你本地帧图像目录
audio_path = "../outputs/sanxingdui1.wav"            # 替换为对应的 .wav 文件
model_weights_path = "model_semi_supervised.pth"       # 替换为你的 .pt 模型文件路径
text_score_map = {}                           # 可暂时为空字典

# ✅ 调用替代函数
keyframes = predict_keyframes_with_onnx(
    frame_dir=frame_dir,
    model_weights_path=model_weights_path,
    audio_path=audio_path,
    text_score_map=text_score_map
)

# ✅ 打印输出
print("关键帧索引与得分：")
for idx, score in keyframes:
    print(f"帧索引: {idx}, 得分: {score}")

