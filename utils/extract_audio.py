import os
from moviepy.editor import VideoFileClip

def extract_audio_from_video(video_path, output_audio_path):
    try:
        video = VideoFileClip(video_path)
        if video.audio:
            video.audio.write_audiofile(output_audio_path, verbose=False, logger=None)
            print(f"[✓] 音频提取完成：{output_audio_path}")
            return output_audio_path
        else:
            print(f"[✗] 无音频轨道：{video_path}")
            return None
    except Exception as e:
        print(f"[✗] 提取音频失败：{video_path}，原因：{e}")
        return None
    finally:
        # 确保资源释放
        if 'video' in locals():
            video.close()

def batch_extract(video_folder, audio_folder):
    os.makedirs(audio_folder, exist_ok=True)
    supported_exts = ['.mp4', '.mov', '.avi', '.mkv']
    failed_files = []
    for file in os.listdir(video_folder):
        if any(file.lower().endswith(ext) for ext in supported_exts):
            input_path = os.path.join(video_folder, file)
            output_path = os.path.join(audio_folder, os.path.splitext(file)[0] + ".wav")
            result = extract_audio_from_video(input_path, output_path)
            if result is None:
                failed_files.append(file)
    if failed_files:
        print("[!] 以下文件提取音频失败：", failed_files)
