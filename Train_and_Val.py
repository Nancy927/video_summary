import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from module5_multimodal_fusion import CNNLSTMAttention
from video_dataset import VideoClipDataset  # 假设你已经实现
from torchvision import models
import csv
import matplotlib.pyplot as plt
import itertools

# ✅ 参数配置
keyframe_root = './outputs/frames'
pseudo_label_file = './outputs/pseudo_labels.json'
epochs = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ✅ MobileNet 作为帧特征提取器
def get_mobilenet_feature_extractor():
    mobilenet = models.mobilenet_v3_small(pretrained=True)
    mobilenet.classifier = nn.Identity()
    mobilenet.eval()
    return mobilenet.to(DEVICE)

# ✅ 提取一个序列内所有帧的特征
def extract_features(model, clip_batch):
    B, T, C, H, W = clip_batch.size()
    clip_batch = clip_batch.view(B * T, C, H, W).to(DEVICE)
    with torch.no_grad():
        features = model(clip_batch)
    return features.view(B, T, -1)

# ✅ 帧级评价指标计算（F1）
def compute_frame_metrics(predictions, labels, threshold=0.5):
    preds = (torch.sigmoid(predictions) >= threshold).int()
    labels = labels.int()
    tp = (preds & labels).sum().item()
    fp = (preds & (~labels.bool())).sum().item()
    fn = ((~preds.bool()) & labels).sum().item()
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1

def train():
    param_grid = {
        'batch_size': [4, 8],
        'learning_rate': [1e-4, 5e-4]
    }
    combinations = list(itertools.product(param_grid['batch_size'], param_grid['learning_rate']))

    result_csv = './outputs/param_results.csv'
    with open(result_csv, 'w', newline='') as rf:
        result_writer = csv.writer(rf)
        result_writer.writerow(['batch_size', 'learning_rate', 'final_val_loss', 'final_f1'])

        for batch_size, learning_rate in combinations:
            print(f"\n🔧 训练参数组合: batch_size={batch_size}, lr={learning_rate}")
            mobilenet = get_mobilenet_feature_extractor()
            model = CNNLSTMAttention().to(DEVICE)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            train_dataset = VideoClipDataset(keyframe_root, pseudo_label_file, split='train')
            val_dataset = VideoClipDataset(keyframe_root, pseudo_label_file, split='val')
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            log_path = f'./outputs/train_log_bs{batch_size}_lr{learning_rate}.csv'
            train_losses, val_losses, f1_scores = [], [], []

            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_precision', 'val_recall', 'val_f1'])

                for epoch in range(epochs):
                    model.train()
                    train_loss = 0.0
                    for clips, labels, _ in train_loader:
                        features = extract_features(mobilenet, clips)
                        labels = labels.float().to(DEVICE)
                        optimizer.zero_grad()
                        outputs = model(features)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item() * clips.size(0)

                    avg_train_loss = train_loss / len(train_dataset)

                    model.eval()
                    val_loss = 0.0
                    all_preds, all_labels = [], []
                    with torch.no_grad():
                        for clips, labels, _ in val_loader:
                            features = extract_features(mobilenet, clips)
                            labels = labels.float().to(DEVICE)
                            outputs = model(features)
                            loss = criterion(outputs, labels)
                            val_loss += loss.item() * clips.size(0)
                            all_preds.append(outputs)
                            all_labels.append(labels)

                    avg_val_loss = val_loss / len(val_dataset)
                    all_preds = torch.cat(all_preds, dim=0)
                    all_labels = torch.cat(all_labels, dim=0)
                    precision, recall, f1 = compute_frame_metrics(all_preds, all_labels)

                    print(f"📘 Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - F1: {f1:.4f}")
                    writer.writerow([epoch+1, avg_train_loss, avg_val_loss, precision, recall, f1])
                    train_losses.append(avg_train_loss)
                    val_losses.append(avg_val_loss)
                    f1_scores.append(f1)

            # 保存模型
            model_path = f'./outputs/model_bs{batch_size}_lr{learning_rate}.pth'
            torch.save(model.state_dict(), model_path)

            # 记录当前参数组合的最终指标
            result_writer.writerow([batch_size, learning_rate, val_losses[-1], f1_scores[-1]])

            # 可视化
            plt.figure(figsize=(10,5))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.plot(f1_scores, label='Val F1')
            plt.xlabel('Epoch')
            plt.ylabel('Loss / F1')
            plt.title(f'batch={batch_size}, lr={learning_rate}')
            plt.legend()
            plt.grid(True)
            fig_path = f'./outputs/curve_bs{batch_size}_lr{learning_rate}.png'
            plt.savefig(fig_path)
            plt.close()

    print("\n✅ 所有训练组合完成，结果保存在 outputs/param_results.csv")

if __name__ == '__main__':
    train()
