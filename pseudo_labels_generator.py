import os
import json
from glob import glob
from PIL import Image

# ====== 配置路径 ======
ALL_FRAMES_DIR = './outputs/frames'              # 所有抽帧图像
KEYFRAMES_DIR = './outputs/keyframes'            # 已选关键帧
PSEUDO_LABELS_FILE = './outputs/pseudo_labels.json'  # 输出文件

# ====== 生成伪标签 ======
def generate_pseudo_labels():
    pseudo_labels = {}

    for video_name in os.listdir(KEYFRAMES_DIR):
        keyframe_folder = os.path.join(KEYFRAMES_DIR, video_name)
        frame_folder = os.path.join(ALL_FRAMES_DIR, video_name)
        
        if not os.path.exists(frame_folder):
            print(f"⚠️ 缺少原始帧目录：{frame_folder}")
            continue

        key_indices = set([
            int(''.join(filter(str.isdigit, os.path.splitext(f)[0])))# 文件名如 000.jpg -> 0
            for f in os.listdir(keyframe_folder) if f.endswith('.jpg')
        ])
        
        all_frame_files = sorted([
            f for f in os.listdir(frame_folder) if f.endswith('.jpg')
        ])
        
        frame_labels = []
        for idx, fname in enumerate(all_frame_files):
            label = 1 if idx in key_indices else 0
            frame_labels.append({
                'frame': fname,
                'label': label
            })
        
        pseudo_labels[video_name] = frame_labels
        print(f"✅ 视频 {video_name} - 共 {len(frame_labels)} 帧，其中正样本 {sum(x['label'] for x in frame_labels)} 帧")

    # 保存 JSON
    with open(PSEUDO_LABELS_FILE, 'w', encoding='utf-8') as f:
        json.dump(pseudo_labels, f, ensure_ascii=False, indent=2)
    print(f"\n📁 已保存伪标签文件：{PSEUDO_LABELS_FILE}")


if __name__ == '__main__':
    generate_pseudo_labels()
