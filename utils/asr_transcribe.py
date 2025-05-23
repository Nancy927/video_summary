import os
import warnings
import whisper
import opencc
import spacy
import json

# 忽略不必要的警告
warnings.filterwarnings("ignore")

# 初始化简体转换器和NER模型（仅初始化一次）
converter = opencc.OpenCC('t2s.json')
nlp = spacy.load("zh_core_web_sm")

# 尝试加载词汇表（避免因路径问题导致模块崩溃）
try:
    current_dir = os.path.dirname(__file__)
    vocab_path = os.path.abspath(os.path.join(current_dir, "sogou_vocab.txt"))
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = set([line.strip() for line in f if line.strip()])
except FileNotFoundError:
    print("[✗] 未找到 sogou_vocab.txt，实体识别提示将不可用")
    vocab = set()

# Whisper 模型仅加载一次（建议 lazy load 或切换 tiny 模型）
try:
    whisper_model = whisper.load_model("base")  # 可替换为 "tiny"
except Exception as e:
    print(f"[✗] Whisper 模型加载失败：{e}")
    whisper_model = None


def convert_to_simplified(text):
    return converter.convert(text)


def highlight_entities(text):
    """
    识别命名实体，提示非词表中的专业名词
    """
    if not vocab:
        return text

    doc = nlp(text)
    for ent in doc.ents:
        if ent.text not in vocab:
            print(f"[术语提示] 可疑术语：{ent.text}")
    return text  # 实际不修改，只提示


def transcribe_audio(audio_path, video_id=None, save_structured=True):
    """
    音频转文字 + 实体提示 + 保存结构化JSON + 返回全文摘要用文本
    """
    if whisper_model is None:
        print("[✗] Whisper 模型未正确初始化")
        return ""

    try:
        result = whisper_model.transcribe(audio_path, language='zh', beam_size=5)
        print(f"[✓] 成功转录: {os.path.basename(audio_path)}")

        segments = result.get("segments", [])
        structured_result = []

        for seg in segments:
            simp_text = convert_to_simplified(seg['text'])
            reviewed_text = highlight_entities(simp_text)
            structured_result.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": reviewed_text
            })

        # === 保存结构化JSON（可选） ===
        if save_structured and video_id:
            output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "transcripts_structured")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{video_id}_structured.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(structured_result, f, ensure_ascii=False, indent=2)
            print(f"[✓] 已保存结构化结果: {output_path}")

        # === 返回拼接文本（供摘要用） ===
        full_text = "。".join([seg["text"] for seg in structured_result])
        return full_text

    except Exception as e:
        print(f"[✗] 转录失败: {audio_path}")
        print(f"原因: {e}")
        return ""
