import os
import numpy as np
import cv2
import torch
from moviepy.editor import VideoFileClip, concatenate_videoclips
from utils.extract_audio import extract_audio_from_video
from utils.module5_multimodal_fusion import (
    extract_frames_per_second,
    extract_visual_features,
    extract_audio_feature,
    compute_final_scores_cnn_lstm_attention,
    remove_similar_frames,
    dynamic_topk_selection,
    select_topkeyframes_by_score,
    CNNLSTMAttention
)

def generate_summary_from_indices(video_path, frame_indices, output_path, clip_margin=2.0, min_gap=2.0):
    """
    根据关键帧编号生成摘要视频（基于帧号直接换算时间戳，自动提取 fps）。
    """
    if not frame_indices:
        raise ValueError("关键帧索引列表为空")

    frame_indices = sorted(set(frame_indices))

    # 自动获取视频帧率
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        raise ValueError("无法从视频中提取有效的 fps")

    # 计算关键帧对应的时间戳
    timestamps = [idx / fps for idx in frame_indices]

    def merge_times_to_segments(times, clip_margin, duration):
        if not times:
            return []
        segments = []
        times = sorted(times)
        start = max(0, times[0] - clip_margin)
        end = min(duration, times[0] + clip_margin)
        for t in times[1:]:
            new_start = max(0, t - clip_margin)
            new_end = min(duration, t + clip_margin)
            if new_start <= end:
                end = max(end, new_end)
            else:
                segments.append((start, end))
                start, end = new_start, new_end
        segments.append((start, end))
        return segments

    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        segments = [video.subclip(start, end) for start, end in merge_times_to_segments(timestamps, clip_margin, duration)]

        if not segments:
            raise ValueError("未生成任何有效摘要片段")

        final_clip = concatenate_videoclips(segments)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 🔧 优化参数避免内存崩溃
        final_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            fps=15,
            bitrate="800k",
            threads=2,
            temp_audiofile='temp-audio.m4a',
            remove_temp=True
        )

        final_clip.close()
        video.close()

    except Exception as e:
        print(f"[ERROR] ❌ 摘要视频合成失败: {e}")
        raise RuntimeError("视频合成过程中发生错误，请检查资源占用或视频内容。")

    # 清理帧图像文件夹
    try:
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        frame_dir = os.path.join('./outputs/frames', video_id)
        if os.path.exists(frame_dir):
            for f in os.listdir(frame_dir):
                file_path = os.path.join(frame_dir, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(frame_dir)
    except Exception as e:
        print(f"[WARN] ❌ 清理帧图像目录失败: {e}")

    return output_path


def extract_frame_timestamps(video_path, keyframe_paths, resize=(224, 224)):
    """
    在原视频中定位关键帧对应时间戳（使用图像匹配）。
    返回每个关键帧的时间（秒）。
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    keyframes = [cv2.resize(cv2.imread(p), resize) for p in keyframe_paths if cv2.imread(p) is not None]
    key_timestamps = []

    for idx in range(frame_count):
        success, frame = cap.read()
        if not success:
            break
        resized = cv2.resize(frame, resize)

        for kf in keyframes:
            diff = np.mean(np.abs(resized.astype(np.float32) - kf.astype(np.float32)))
            if diff < 10:  # 匹配阈值可调
                t = idx / fps
                key_timestamps.append(t)
                keyframes.remove(kf)  # 匹配成功就移除
                break

        if not keyframes:
            break

    cap.release()
    return sorted(key_timestamps)

def get_summary_frame_indices(video_path, audio_path, text_score_map,
                              model_weights_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    整合视觉、音频、文本得分，返回关键帧的索引编号列表。
    参数:
        video_path: 视频文件路径
        audio_path: 音频文件路径（.wav）
        text_score_map: dict[str, float]，视频 ID -> 文本得分（例如 TextRank 输出）
        model_weights_path: 模型 .pth 文件路径
        device: 默认自动使用 GPU/CPU
    返回:
        List[int]，关键帧索引
    """
    print("[DEBUG] ✅ 正在使用 get_summary_frame_indices 模型推理...")
    # 加载注意力模型
    model = CNNLSTMAttention().to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    # 每秒提取一帧图像
    raw_images, frame_indices = extract_frames_per_second(video_path)  # 新方法

    # 提取视觉特征
    visual_feats = extract_visual_features(raw_images)  # shape: (1, N, 576)

    # 提取音频特征
    audio_feat = extract_audio_feature(audio_path)  # shape: (64,)

    # 文本得分（从 map 中获取 video_id）
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    text_score_scalar = text_score_map.get(video_id, 0.5)

    # 融合打分（视觉注意力输出 + 音频相似度 + 文本得分）
    fused_scores = compute_final_scores_cnn_lstm_attention(
        visual_seq=visual_feats,
        audio_feat=audio_feat,
        text_score_scalar=text_score_scalar
    )

    # 相似帧去重（返回图像、索引、特征）
    filtered_images, keep_indices, filtered_feats = remove_similar_frames(
        raw_images, visual_feats, threshold=0.92
    )

    # 根据融合得分动态选择 K
    filtered_scores = fused_scores[keep_indices]
    topk = dynamic_topk_selection(filtered_scores, min_frames=10, max_frames=30)
    selected_indices = select_topkeyframes_by_score(filtered_scores, topk=topk)

    # 映射回原始帧索引（以便后续推算时间戳）
    final_indices = [frame_indices[keep_indices[i]] for i in selected_indices]  # 关键修改点

    return final_indices
