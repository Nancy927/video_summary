import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, models
import onnxruntime as ort
from utils.video_summarizer import get_summary_frame_indices

# ✅ 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ✅ 加载 ONNX 模型
def load_onnx_model(onnx_path):
    return ort.InferenceSession(onnx_path)

# ✅ 读取帧图像路径（按顺序）
def load_frame_paths(frame_dir):
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    return [os.path.join(frame_dir, f) for f in frame_files]

# ✅ 提取 clip 特征序列（每个clip含 T 帧）
def extract_clip_batches(frame_paths, clip_len=5, stride=1):
    clips = []
    indices = []
    for i in range(0, len(frame_paths) - clip_len + 1, stride):
        clip = frame_paths[i:i+clip_len]
        clips.append(clip)
        indices.append(i + clip_len // 2)  # 取中间帧索引作为代表帧
    return clips, indices

# ✅ 对单个 clip 提取预处理后的 Tensor
def process_clip(clip_paths):
    clip_images = [preprocess(Image.open(p).convert('RGB')) for p in clip_paths]
    return torch.stack(clip_images).unsqueeze(0).numpy()  # (1, T, 3, 224, 224)

# ✅ 主函数：输入帧目录，输出关键帧得分列表
def predict_keyframes_with_onnx(frame_dir, model_weights_path, audio_path=None, text_score_map=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    替代 ONNX 推理：调用 get_summary_frame_indices 执行多模态融合，返回关键帧索引与虚拟得分（score=1.0）。
    返回值格式保持与旧版相同：[(index1, score1), (index2, score2), ...]
    """
    import cv2

    print(f"[INFO] 🚀 使用本地模型推理替代 ONNX: {model_weights_path}")
    # 获取帧编号 → 视频路径
    frame_paths = load_frame_paths(frame_dir)
    if not frame_paths:
        raise ValueError("❌ 帧目录为空")
    
    # 根据帧路径合成 video_path
    frame_dir_path = os.path.abspath(frame_dir)
    video_path = os.path.join(os.path.dirname(frame_dir_path), "temp_input.mp4")

    # 使用 OpenCV 合成视频文件，便于调用 video_summarizer 接口
    fps = 30
    frame = cv2.imread(frame_paths[0])
    h, w, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    for path in frame_paths:
        img = cv2.imread(path)
        out.write(img)
    out.release()
    print(f"[INFO] ✅ 合成临时视频完成: {video_path}")

    # 执行替代推理
    key_indices = get_summary_frame_indices(
        video_path=video_path,
        audio_path=audio_path,
        text_score_map=text_score_map or {},
        model_weights_path=model_weights_path,
        device=device
    )

    print(f"[INFO] ✅ 得到关键帧索引数量: {len(key_indices)}")

    return [(idx, 1.0) for idx in key_indices]  # 保持接口一致