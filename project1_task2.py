import os
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, det_curve
import python_speech_features as psf
import wave
import joblib
from scipy.signal import butter, lfilter
from evaluate import *
import re

class SJTUVoiceActivityDetector:
    def __init__(self, wav_dir=None, label_file=None, feature_type='mfcc',
                 n_mfcc=13, n_fft=512, hop_length=160, sr=16000,
                 gmm_components=16, preemphasis=True, bandpass=True):
        """初始化VAD检测器"""
        self.wav_dir = wav_dir
        self.label_file = label_file
        self.feature_type = feature_type
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.gmm_components = gmm_components
        self.preemphasis = preemphasis
        self.bandpass = bandpass

        # GMM模型初始化
        self.speech_model = GaussianMixture(n_components=gmm_components,
                                            covariance_type='diag',
                                            max_iter=100, random_state=42)
        self.nonspeech_model = GaussianMixture(n_components=gmm_components,
                                               covariance_type='diag',
                                               max_iter=100, random_state=42)

    def apply_preemphasis(self, audio, coefficient=0.97):
        """预加重滤波器"""
        return np.append(audio[0], audio[1:] - coefficient * audio[:-1])

    def butter_bandpass(self, lowcut=300, highcut=8000, order=5):
        """带通滤波器设计"""
        nyq = 0.5 * self.sr
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def bandpass_filter(self, data, lowcut=300, highcut=8000):
        """应用带通滤波"""
        b, a = self.butter_bandpass(lowcut, highcut)
        return lfilter(b, a, data)

    def load_audio(self, filename, wav_dir=None):
        """加载并预处理音频"""
        wav_dir = wav_dir or self.wav_dir
        filepath = os.path.join(wav_dir, filename + ".wav")

        with wave.open(filepath, 'rb') as wav:
            n_channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            framerate = wav.getframerate()
            audio_data = np.frombuffer(wav.readframes(wav.getnframes()),
                                       dtype=np.int16 if sampwidth == 2 else np.int8)

        # 转单声道
        if n_channels > 1:
            audio_data = np.mean(audio_data.reshape(-1, n_channels), axis=1)

        # 预加重
        if self.preemphasis:
            audio_data = self.apply_preemphasis(audio_data)

        # 带通滤波
        if self.bandpass and self.sr > 8000 * 2:  # 满足Nyquist定理
            audio_data = self.bandpass_filter(audio_data)

        return audio_data, framerate

    def extract_features(self, audio_data):
        """提取MFCC特征"""
        mfcc_feat = psf.mfcc(audio_data, samplerate=self.sr,
                             winlen=self.n_fft / self.sr,
                             winstep=self.hop_length / self.sr,
                             numcep=self.n_mfcc,
                             nfilt=40, nfft=self.n_fft)
        delta = psf.delta(mfcc_feat, 2)
        delta_delta = psf.delta(delta, 2)
        return np.hstack([mfcc_feat, delta, delta_delta])

    def parse_labels(self, label_file=None):
        """解析标签文件"""
        label_file = label_file or self.label_file
        label_dict = {}
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        audio_id = parts[0]
                        segments = [tuple(map(float, seg.split(','))) for seg in parts[1:]]
                        label_dict[audio_id] = segments
        return label_dict

    def load_labeled_data(self, wav_dir=None, label_file=None, file_list=None):
        """加载带标签的数据"""
        wav_dir = wav_dir or self.wav_dir
        label_file = label_file or self.label_file

        X, y = [], []
        label_dict = self.parse_labels(label_file)
        files_to_process = file_list if file_list else label_dict.keys()

        for audio_id in files_to_process:
            try:
                audio_data, sr = self.load_audio(audio_id, wav_dir)
                if sr != self.sr:
                    continue

                features = self.extract_features(audio_data)
                frame_times = np.arange(len(features)) * (self.hop_length / self.sr)
                frame_labels = np.zeros(len(features), dtype=int)

                for start, end in label_dict.get(audio_id, []):
                    mask = (frame_times >= start) & (frame_times <= end)
                    frame_labels[mask] = 1

                X.append(features)
                y.append(frame_labels)
            except Exception as e:
                print(f"Error processing {audio_id}: {str(e)}")
                continue

        if not X:
            raise ValueError("No valid data loaded")
        return np.concatenate(X), np.concatenate(y)

    def train(self, X_train=None, y_train=None, train_dir=None, train_label=None):
        """训练模型"""
        if X_train is None or y_train is None:
            if train_dir and train_label:
                X_train, y_train = self.load_labeled_data(wav_dir=train_dir, label_file=train_label)
            else:
                X_train, y_train = self.load_labeled_data()

        print(f"Training data - Speech: {sum(y_train)}, Non-speech: {len(y_train) - sum(y_train)}")

        # 训练模型
        print("Training speech model...")
        self.speech_model.fit(X_train[y_train == 1])
        print("Training non-speech model...")
        self.nonspeech_model.fit(X_train[y_train == 0])

        # 验证训练效果
        train_pred = self.predict(X_train)
        print("\nTraining set performance:")
        print(classification_report(y_train, train_pred, target_names=['non-speech', 'speech']))

    def predict(self, features):
        """预测帧级标签"""
        speech_scores = self.speech_model.score_samples(features)
        nonspeech_scores = self.nonspeech_model.score_samples(features)
        return (speech_scores > nonspeech_scores).astype(int)

    def evaluate(vad, X_test, y_test):
        y_pred_prob = np.exp(vad.speech_model.score_samples(X_test)) / (
                np.exp(vad.speech_model.score_samples(X_test)) + np.exp(vad.nonspeech_model.score_samples(X_test)))
        y_pred = (y_pred_prob > 0.5).astype(int)

        # 计算准确率
        accuracy = np.sum(y_pred == y_test) / len(y_test)

        # 计算ROC曲线和AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # 计算EER
        fpr_det, fnr_det, thresholds_det = det_curve(y_test, y_pred_prob)
        eer = fpr_det[np.nanargmin(np.absolute((fnr_det - fpr_det)))]

        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {roc_auc:.4f}")
        print(f"EER: {eer:.4f}")

        print(classification_report(y_test, y_pred, target_names=['non-speech', 'speech']))

        return y_pred
    def save_model(self, model_path):
        """保存模型"""
        joblib.dump({
            'speech_model': self.speech_model,
            'nonspeech_model': self.nonspeech_model,
            'config': {
                'feature_type': self.feature_type,
                'n_mfcc': self.n_mfcc,
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'sr': self.sr,
                'gmm_components': self.gmm_components,
                'preemphasis': self.preemphasis,
                'bandpass': self.bandpass
            }
        }, model_path)

    @classmethod
    def load_model(cls, model_path):
        """加载模型"""
        model_data = joblib.load(model_path)
        vad = cls(**model_data['config'])
        vad.speech_model = model_data['speech_model']
        vad.nonspeech_model = model_data['nonspeech_model']
        return vad

    def return_segment(self, file, wav):
        audio_data, sr = self.load_audio(file, wav)
        features = self.extract_features(audio_data)

        speech_probs = np.exp(self.speech_model.score_samples(features))
        nonspeech_probs = np.exp(self.nonspeech_model.score_samples(features))
        y_pred = (speech_probs > nonspeech_probs).astype(int)

        frame_times = np.arange(len(y_pred)) * (self.hop_length / self.sr)
        segments = []
        in_segment = False
        start_time = 0
        for i, label in enumerate(y_pred):
            if label == 1 and not in_segment:
                start_time = frame_times[i]
                in_segment = True
            elif label == 0 and in_segment:
                segments.append((start_time, frame_times[i]))
                in_segment = False

        # 处理最后一个段
        if in_segment:
            segments.append((start_time, frame_times[-1]))

        return segments


def process_audio_files(input_dir, output_file, vad):
    """
    处理输入目录下的所有wav文件，保存端点检测结果到文本文件
    :param input_dir: 输入wav文件目录
    :param output_file: 输出文本文件路径
    """
    total_time = 20.0  # 音频总时长（秒）
    sr = 100  # 采样率（Hz）
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # 遍历目录中的所有wav文件
        for filename in os.listdir(input_dir):
            print(filename)
            if filename.endswith('.wav'):
                # 进行端点检测
                new_filename = filename.replace(".wav", "")
                segments = vad.return_segment(new_filename, input_dir)

                # segments_dict = build_audio_segments_dict("voice-activity-detection-sjtu-spring-2024\\vad\data\dev_label.txt")
                #
                # y_pred = time_to_frames(segments, total_time, sr)
                # y_true = time_to_frames(segments_dict[new_filename], total_time, sr)
                #
                # acc = accuracy_score(y_true, y_pred)
                # fpr, tpr, thresholds = roc_curve(y_true, y_pred)
                # auc = roc_auc_score(y_true, y_pred)
                # fnr = 1 - tpr
                # eer = fpr[np.nanargmin(np.abs(fnr - fpr))]

                # # 写入结果
                # f_out.write(f"File: {filename}\n")
                # f_out.write(f"Segment: Frame [{segments}]\n")
                # # f_out.write(f"Evaluate: {get_metrics(segments,segments_dict[new_filename])}\n")
                # f_out.write(f"Evaluate: Accuracy:{acc}\n"
                #             f"AUC:{auc}\n"
                #             f"EER:{eer}\n")
                # f_out.write("\n")  # 文件间空行分隔

                # 写入结果
                f_out.write(f"{filename}\t")
                f_out.write(f"{segments}\n")
                # f_out.write(f"Evaluate: {get_metrics(segments,segments_dict[new_filename])}\n")

def clean_timestamps_file(input_file, output_file, decimal_places=2):
    """
    处理 input.txt，规范化时间戳的浮点数精度
    :param input_file: 输入文件名（如 "input.txt"）
    :param output_file: 输出文件名（如 "output.txt"）
    :param decimal_places: 保留的小数位数（默认 2）
    """
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue  # 跳过空行

            # 分割文件名和时间戳部分
            parts = line.split('\t')
            if len(parts) != 2:
                f_out.write(line + '\n')  # 如果格式错误，直接写入原行
                continue

            filename, timestamps = parts[0], parts[1]

            # 处理时间戳
            cleaned_pairs = []
            for pair in timestamps.split():
                try:
                    start, end = pair.split(',')
                    # 四舍五入到指定小数位
                    start_clean = round(float(start), decimal_places)
                    end_clean = round(float(end), decimal_places)
                    # 格式化为字符串，避免科学计数法（如 1.0 -> "1.0" 而不是 "1"）
                    start_str = format(start_clean, f".{decimal_places}f")
                    end_str = format(end_clean, f".{decimal_places}f")
                    cleaned_pairs.append(f"{start_str},{end_str}")
                except ValueError:
                    cleaned_pairs.append(pair)  # 如果拆分失败，保留原数据

            # 重新组合并写入文件
            cleaned_timestamps = ' '.join(cleaned_pairs)
            f_out.write(f"{filename}\t{cleaned_timestamps}\n")

def merge_short_intervals(timestamps, min_silence=0.3, min_voice=0.1):
    if not timestamps:
        return []

    # 初始化合并后的列表
    merged = [list(map(float, timestamps[0].split(',')))]

    for pair in timestamps[1:]:
        start, end = map(float, pair.split(','))
        last_end = merged[-1][1]

        # 检查非语音段是否过短
        if (start - last_end) < min_silence:
            # 合并语音段
            merged[-1][1] = end
        else:
            # 检查语音段是否过短
            if (end - start) >= min_voice:
                merged.append([start, end])
            # 否则跳过（视为噪声）

    return merged


def process_file(input_file, output_file, min_silence=0.3, min_voice=0.1):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if not line or '\t' not in line:
                f_out.write(line + '\n')
                continue

            filename, timestamps = line.split('\t', 1)
            pairs = timestamps.split()

            # 合并过短间隔
            merged_pairs = merge_short_intervals(pairs, min_silence, min_voice)

            # 格式化输出
            cleaned_timestamps = ' '.join([f"{s},{e}" for s, e in merged_pairs])
            f_out.write(f"{filename}\t{cleaned_timestamps}\n")

if __name__ == "__main__":
    # 配置路径
    base_dir = "voice-activity-detection-sjtu-spring-2024/vad"
    train_config = {
        'wav_dir': os.path.join(base_dir, "wavs", "train"),
        'label_file': os.path.join(base_dir, "data", "train_label.txt"),
        'preemphasis': True,
        'bandpass': True,
        'n_mfcc': 13,
        'sr': 16000,
        'gmm_components': 8
    }
    model_path = "vad_model.joblib"

    # 训练或加载模型
    tra_wav = os.path.join(base_dir, "wavs", "train")
    tra_label = os.path.join(base_dir, "data", "train_label.txt")
    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        vad = SJTUVoiceActivityDetector.load_model(model_path)
        X_tra, y_tra = vad.load_labeled_data(wav_dir=tra_wav, label_file=tra_label)
        print("\nTraining Set Evaluation:")
        vad.evaluate(X_tra, y_tra)
    else:
        print("Training new model...")
        vad = SJTUVoiceActivityDetector(**train_config)
        vad.train()
        vad.save_model(model_path)
        X_tra, y_tra = vad.load_labeled_data(wav_dir=tra_wav, label_file=tra_label)
        print("\nTraining Set Evaluation:")
        vad.evaluate(X_tra, y_tra)
        print(f"Model saved to {model_path}")

    # 开发集评估
    dev_wav = os.path.join(base_dir, "wavs", "dev")
    dev_label = os.path.join(base_dir, "data", "dev_label.txt")
    try:
        X_dev, y_dev = vad.load_labeled_data(wav_dir=dev_wav, label_file=dev_label)
        print("\nDevelopment Set Evaluation:")
        vad.evaluate(X_dev, y_dev)
    except Exception as e:
        print(f"Dev set evaluation error: {str(e)}")

    # 测试单个文件
    test_files = ["5244-54280-0021", "54-121080-0009"]
    for file in test_files:
        try:
            test_path = os.path.join(dev_wav, file + ".wav")
            if os.path.exists(test_path):
                X_test, y_test = vad.load_labeled_data(wav_dir=dev_wav, label_file=dev_label, file_list=[file])
                print(f"\nEvaluating {file}:")
                vad.evaluate(X_test, y_test)
        except Exception as e:
            print(f"Error testing {file}: {str(e)}")

    # 测试集上面进行
    # 输入输出路径设置
    input_dir = r"voice-activity-detection-sjtu-spring-2024\vad\wavs\test"
    output_file = "vad_test_results_task2.txt"

    # 确保输入目录存在
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    # 处理所有音频文件
    process_audio_files(input_dir, output_file, vad)

    with open('vad_test_results_task2.txt', 'r') as f_in, open('vad_test_results_task2_new.txt', 'w') as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            # 分割文件名和时间数据
            filename, times = line.split('\t', 1)

            # 去除.wav后缀
            filename = filename.replace('.wav', '')

            # 使用正则提取所有(x, y)对
            pairs = re.findall(r'\(([^)]+)\)', times)

            # 处理每个对：去除空格，替换逗号后的空格
            processed_pairs = []
            for pair in pairs:
                cleaned = pair.replace(' ', '').replace(',', ',')  # 确保格式一致
                processed_pairs.append(cleaned)

            # 组合成目标格式
            new_times = ' '.join(processed_pairs)

            f_out.write(f"{filename}\t{new_times}\n")

    # 调用函数处理文件
    clean_timestamps_file("vad_test_results_task2_new.txt", "vad_test_results_task2_new_new.txt", decimal_places=2)

    # 示例调用
    process_file("vad_test_results_task2_new_new.txt", "vad_test_results_task2.txt", min_silence=0.3, min_voice=0.1)

    print(f"处理完成，结果已保存到 {output_file}")

