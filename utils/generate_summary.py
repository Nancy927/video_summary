# generate_summary.py
import os
import json
import torch
import numpy as np
from transformers import BertTokenizer, BartForConditionalGeneration
import onnxruntime as ort
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

# ===== 模型路径配置 =====
BART_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "bart-base-chinese")
ONNX_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "keyframe_model_simplified.onnx")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lazy-load BART
_tokenizer = None
_bart_model = None


def load_bart():
    global _tokenizer, _bart_model
    if _tokenizer is None or _bart_model is None:
        print("[加载] 初始化 BART 模型...")
        _tokenizer = BertTokenizer.from_pretrained(BART_MODEL_PATH)
        _bart_model = BartForConditionalGeneration.from_pretrained(BART_MODEL_PATH).to(DEVICE)
    return _tokenizer, _bart_model


# 图像预处理
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def merge_segments(segments, group_size=5):
    merged = []
    i = 0
    n = len(segments)
    while i < n:
        start = segments[i]["start"]
        end = segments[i]["end"]
        texts = []
        for j in range(i, min(i + group_size, n)):
            texts.append(segments[j]["text"].strip())
            end = segments[j]["end"]
        merged.append({
            "start": start,
            "end": end,
            "text": "。".join(texts)
        })
        i += group_size
    return merged


def split_text(text, max_tokens=900):
    tokenizer, _ = load_bart()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_ids = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += max_tokens
    return chunks


def generate_summary(text, max_input_len=1024, max_output_len=128, min_output_len=8):
    tokenizer, bart_model = load_bart()
    prompt = "请用一句话概括："
    chunks = split_text(text, max_tokens=900)
    all_summaries = []

    for chunk in chunks:
        input_text = prompt + chunk
        inputs = tokenizer(input_text, return_tensors="pt", max_length=max_input_len, truncation=True, padding="max_length").to(DEVICE)
        with torch.no_grad():
            summary_ids = bart_model.generate(
                inputs["input_ids"],
                max_length=max_output_len,
                min_length=min_output_len,
                num_beams=5,
                length_penalty=2.0,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        all_summaries.append(summary)

    print(f"[✓] 文本摘要完成，共 {len(all_summaries)} 段")
    return " / ".join(all_summaries)


def select_keyframes(frame_dir, onnx_model_path, top_k=3):
    try:
        ort_session = ort.InferenceSession(onnx_model_path)
        input_name = ort_session.get_inputs()[0].name
    except Exception as e:
        print(f"[✗] ONNX模型加载失败：{e}")
        return []

    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(('.jpg', '.png'))])
    image_paths = [os.path.join(frame_dir, f) for f in frame_files]

    images = []
    valid_paths = []

    for path in image_paths:
        try:
            img = image_transform(Image.open(path).convert("RGB"))
            images.append(img)
            valid_paths.append(path)
        except (UnidentifiedImageError, OSError) as e:
            print(f"[跳过] 无法加载图像: {path}，原因: {e}")

    if not images:
        print("[✗] 没有可用图像帧")
        return []

    img_tensor = torch.stack(images).numpy().astype(np.float32)

    try:
        outputs = ort_session.run(None, {input_name: img_tensor})
        scores = outputs[0].squeeze()
    except Exception as e:
        print(f"[✗] ONNX 推理失败: {e}")
        return []

    keyframes = list(zip(valid_paths, scores))
    keyframes.sort(key=lambda x: x[1], reverse=True)
    selected = [path for path, _ in keyframes[:top_k]]

    print(f"[✓] 已选取关键帧: {selected}")
    return selected


def generate_summary_and_keyframes(video_id):
    transcript_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "transcripts_structured", f"{video_id}_structured.json")
    frame_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "frames", video_id)

    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"❌ 未找到转录文件：{transcript_path}")
    if not os.path.exists(frame_dir):
        raise FileNotFoundError(f"❌ 未找到帧目录：{frame_dir}")

    with open(transcript_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    merged = merge_segments(segments, group_size=5)
    full_text = "。".join([seg["text"] for seg in merged])
    summary = generate_summary(full_text)
    keyframe_paths = select_keyframes(frame_dir, ONNX_MODEL_PATH, top_k=3)

    return summary, keyframe_paths


def generate_summary_text(transcript_text: str):
    return generate_summary(transcript_text)