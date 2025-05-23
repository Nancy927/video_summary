import os
import re
import cv2
import json
import numpy as np
import torch
import librosa
import jieba.analyse
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

# ====== 路径配置 ======
VIDEO_DIR = './videos'
AUDIO_DIR = './outputs/audio'
FEATURE_DIR = './outputs/features'
KEYFRAME_DIR = './outputs/keyframes'
CAPTION_FILE = './outputs/auto_captions.json'
TEXT_SCORE_FILE = './outputs/text_scores.json'

os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(KEYFRAME_DIR, exist_ok=True)

# ====== 设备设置 ======
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\U0001F4E6 当前设备：{device}")

# ====== MobileNetV3 视觉特征提取器 ======
mobilenet_model = torch.nn.Sequential(
    models.mobilenet_v3_small(pretrained=True).features,
    torch.nn.AdaptiveAvgPool2d((1, 1))
).eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== 线性投影层 ======
class VisualProjector(nn.Module):
    def __init__(self, input_dim=576, output_dim=64):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)

visual_projector = VisualProjector().to(device).eval()

# ===== 简化版 VGGish 模型 =====
class SimpleVGGish(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.conv(x)
        return x.view(x.size(0), -1)  # shape: (B, 64)

vggish_model = SimpleVGGish().to('cpu').eval()
mel_transform = MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=64)
db_transform = AmplitudeToDB()

# ===== CNN + LSTM + Attention 模型结构 =====
class CNNLSTMAttention(nn.Module):
    def __init__(self, input_dim=576, hidden_dim=128, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attn_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attn_fc(lstm_out), dim=1)
        weighted = lstm_out * attn_weights
        scores = weighted.sum(dim=2).squeeze(0)
        return scores

cnn_lstm_att_model = CNNLSTMAttention().to(device).eval()

# ===== 文本摘要得分加载（TextRank） =====
def preprocess_text(text):
    return re.sub(r'[^\u4e00-\u9fff0-9a-zA-Z。，！？]', '', text)

def load_text_scores_textrank():
    with open(CAPTION_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    text_scores = {}
    for vid, segments in data.items():
        raw_text = " ".join([s.get("segment", "") for s in segments]) if isinstance(segments, list) else ""
        keywords = jieba.analyse.textrank(preprocess_text(raw_text), topK=10, withWeight=True)
        score = sum(w for _, w in keywords)
        text_scores[vid] = round(score, 4)
    max_score = max(text_scores.values(), default=1.0)
    for k in text_scores:
        text_scores[k] = text_scores[k] / max_score if max_score else 0.5
    with open(TEXT_SCORE_FILE, 'w', encoding='utf-8') as f:
        json.dump(text_scores, f, ensure_ascii=False, indent=2)
    return text_scores

text_score_map = load_text_scores_textrank()

# ===== 视频帧提取 =====
def extract_frames_per_second(video_path):
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps)  # 每秒一帧

    frames = []
    indices = []

    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            indices.append(idx)
        idx += 1
    cap.release()
    return frames, indices


# ===== 视觉特征提取 =====
def extract_visual_features(frames):
    feats = []
    with torch.no_grad():
        for frame in tqdm(frames, desc="\U0001F3A8 提取视觉特征"):
            input_tensor = transform(frame).unsqueeze(0).to(device)
            feat = mobilenet_model(input_tensor).squeeze().flatten()
            feats.append(feat)
    return torch.stack(feats).unsqueeze(0).to(device)

# ===== 音频特征提取 =====
def extract_audio_feature(wav_path):
    y, _ = librosa.load(wav_path, sr=16000)
    y = torch.tensor(y).float()
    mel = mel_transform(y)
    mel_db = db_transform(mel).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        feat = vggish_model(mel_db.to('cpu'))
    return feat.squeeze().cpu().numpy()

# ===== 相似帧过滤 =====
def remove_similar_frames(frames, features, threshold=0.92):
    features_np = features.squeeze(0).cpu().numpy()
    keep_indices = [0]
    last_feat = features_np[0]
    for i in range(1, len(features_np)):
        sim = cosine_similarity([last_feat], [features_np[i]])[0][0]
        if sim < threshold:
            keep_indices.append(i)
            last_feat = features_np[i]
    return [frames[i] for i in keep_indices], keep_indices, features[:, keep_indices, :]

# ===== 动态关键帧数量决策 =====
def dynamic_topk_selection(scores, min_frames=10, max_frames=30):
    scores = np.array(scores)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    threshold = mean_score + 0.2 * std_score
    dynamic_k = np.sum(scores >= threshold)
    return np.clip(dynamic_k, min_frames, max_frames)

# ===== 融合得分计算 =====
def compute_final_scores_cnn_lstm_attention(visual_seq, audio_feat, text_score_scalar,
                                            weight_v=0.5, weight_a=0.3, weight_t=0.2):
    n_frames = visual_seq.shape[1]
    with torch.no_grad():
        visual_score = cnn_lstm_att_model(visual_seq).cpu().numpy()
        visual_proj = visual_projector(visual_seq.squeeze(0)).cpu().numpy()
    audio_vec = np.tile(audio_feat.reshape(1, -1), (n_frames, 1))
    audio_score = cosine_similarity(visual_proj, audio_vec).diagonal()
    text_score = np.full(n_frames, text_score_scalar)
    visual_score = visual_score / (np.linalg.norm(visual_score) or 1.0)
    audio_score = audio_score / (np.linalg.norm(audio_score) or 1.0)
    text_score = text_score / (np.linalg.norm(text_score) or 1.0)

    final_scores = weight_v * visual_score + weight_a * audio_score + weight_t * text_score
    return final_scores

# ===== Top-K 关键帧选择 =====
def select_topkeyframes_by_score(scores, topk=8):
    topk = min(topk, len(scores))
    sorted_indices = np.argsort(scores)[::-1][:topk]
    return sorted(sorted_indices)