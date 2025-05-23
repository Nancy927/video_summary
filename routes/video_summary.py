# routes/video_summary.py

from flask import Blueprint, request, jsonify
import os
import uuid
import shutil
import traceback  # ✅ 用于打印详细错误堆栈
from werkzeug.utils import secure_filename
from utils.extract_audio import extract_audio_from_video
from utils.asr_transcribe import transcribe_audio
from utils.generate_summary import generate_summary_text
from utils.mobilenet_feat import predict_keyframes_with_onnx
from utils.video_summarizer import generate_summary_from_indices
import numpy as np
import cv2

video_summary_api = Blueprint('video_summary_api', __name__)

UPLOAD_FOLDER = './temp_uploads'
FRAME_FOLDER = './outputs/frames'
AUDIO_FOLDER = './outputs/audio'
SUMMARY_FOLDER = './static/outputs/summ_videos'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

torch_model_path = './utils/model_semi_supervised.pth'


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@video_summary_api.route('/video_summary', methods=['POST'])
def video_summary():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video part'}), 400

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if video_file and allowed_file(video_file.filename):
            # 1. 保存上传视频
            uid = str(uuid.uuid4())
            filename = secure_filename(video_file.filename)
            video_path = os.path.join(UPLOAD_FOLDER, f'{uid}_{filename}')
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            video_file.save(video_path)

            if not os.path.exists(video_path):
                raise FileNotFoundError(f"❌ 视频保存失败: {video_path}")

            # 2. 抽取音频并生成摘要文本
            os.makedirs(AUDIO_FOLDER, exist_ok=True)
            audio_path = os.path.join(AUDIO_FOLDER, f'{uid}.wav')
            extract_audio_from_video(video_path, audio_path)

            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"❌ 音频文件未生成: {audio_path}")

            transcript = transcribe_audio(audio_path)
            summary_text = generate_summary_text(transcript)

            # 3. 抽帧
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_dir = os.path.join(FRAME_FOLDER, uid)
            os.makedirs(frame_dir, exist_ok=True)

            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_path = os.path.join(frame_dir, f'{frame_index:05d}.jpg')
                cv2.imwrite(frame_path, frame)
                frame_index += 1
            cap.release()

            if frame_index == 0:
                raise RuntimeError(f"❌ 视频帧提取失败: {video_path}")

            # 4. 关键帧筛选
            print("[DEBUG] 使用本地模型进行关键帧预测（替代 ONNX）...")
            scores = predict_keyframes_with_onnx(
                frame_dir=frame_dir,
                model_weights_path=torch_model_path,
                audio_path=audio_path
            )

            print(f"[DEBUG] 模型返回得分数量: {len(scores)}，示例分数: {scores[:5]}")

            if not scores:
                raise RuntimeError("❌ 模型未返回任何帧得分")

            # 筛选得分最高的 10% 帧
            all_scores = [score for _, score in scores]
            threshold = np.percentile(all_scores, 90)
            keyframe_indices = [index for index, score in scores if score >= threshold]

            if not keyframe_indices:
                raise RuntimeError("❌ 未检测到关键帧")

            # 5. 生成摘要视频
            os.makedirs(SUMMARY_FOLDER, exist_ok=True)
            output_filename = f'{uid}_summary.mp4'
            output_video_path = os.path.join(SUMMARY_FOLDER, output_filename)

            generate_summary_from_indices(
                video_path=video_path,
                frame_indices=keyframe_indices,
                output_path=output_video_path,
                clip_margin=2.0,
                min_gap=3.0
            )

            if not os.path.exists(output_video_path):
                raise FileNotFoundError(f"❌ 摘要视频生成失败: {output_video_path}")
            
            # ✅ 删除中间音频文件和帧图像文件
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(frame_dir):
                shutil.rmtree(frame_dir)

            # 6. 返回摘要文本 + 视频路径
            return jsonify({
                "summary_text": summary_text,
                "summary_video_path": f"/static/outputs/summ_videos/{output_filename}"
            })
        
        

        else:
            return jsonify({'error': 'Invalid file format'}), 400

    except Exception as e:
        traceback.print_exc()  # ✅ 输出详细错误堆栈到终端

        return jsonify({'error': str(e)}), 500
