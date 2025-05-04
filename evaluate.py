from sklearn import metrics
import numpy as np
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt


def compute_eer(target_scores, nontarget_scores):
    """Calculate EER following the same way as in Kaldi.

    Args:
        target_scores (array-like): sequence of scores where the
                                    label is the target class
        nontarget_scores (array-like): sequence of scores where the
                                    label is the non-target class
    Returns:
        eer (float): equal error rate
        threshold (float): the value where the target error rate
                           (the proportion of target_scores below
                           threshold) is equal to the non-target
                           error rate (the proportion of nontarget_scores
                           above threshold)
    """
    assert len(target_scores) != 0 and len(nontarget_scores) != 0
    tgt_scores = sorted(target_scores)
    nontgt_scores = sorted(nontarget_scores)

    target_size = float(len(tgt_scores))
    nontarget_size = len(nontgt_scores)
    target_position = 0
    for target_position, tgt_score in enumerate(tgt_scores[:-1]):
        nontarget_n = nontarget_size * target_position / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)
        if nontarget_position < 0:
            nontarget_position = 0
        if nontgt_scores[nontarget_position] < tgt_score:
            break
    threshold = tgt_scores[target_position]
    eer = target_position / target_size
    return eer, threshold


def get_metrics(prediction, label):
    """Calculate several metrics for a binary classification task.

    Args:
        prediction (array-like): sequence of probabilities
            e.g. [0.1, 0.4, 0.35, 0.8]
        labels (array-like): sequence of class labels (0 or 1)
            e.g. [0, 0, 1, 1]
    Returns:
        auc: area-under-curve
        eer: equal error rate
    """  # noqa: H405, E261
    assert len(prediction) == len(label), (len(prediction), len(label))
    fpr, tpr, thresholds = metrics.roc_curve(label, prediction, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # from scipy.optimize import brentq
    # from scipy.interpolate import interp1d
    # fnr = 1 - tpr
    # eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    eer, thres = compute_eer(
        [pred for i, pred in enumerate(prediction) if label[i] == 1],
        [pred for i, pred in enumerate(prediction) if label[i] == 0],
    )
    return auc, eer


# 1. 数据预处理
def time_to_frames(segments, total_time, sr=100):
    frames = np.zeros(int(total_time * sr), dtype=int)
    for i in range(0, len(segments), 2):
        start = int(segments[i] * sr)
        end = int(segments[i+1] * sr)
        frames[start:end] = 1
    return frames

# if __name__ == "__main__":
#     # 第一个参数为模型预测输出（可以是概率，也可以是二值分类结果）
#     # 第二个参数为数据对应的标签
#     print(get_metrics([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]))
#     # 注意：计算最终指标时，应将整个数据集（而不是在每个样本上单独计算）的所有语音帧预测结果合并在一个list中，
#     # 对应的标签也合并在一个list中，然后再调用get_metrics来计算指标

# import numpy as np
# from sklearn.metrics import roc_curve, auc, accuracy_score
# import matplotlib.pyplot as plt
#
# # 假设帧长为25ms
# frame_length_ms = 25
# frame_length_sec = frame_length_ms / 1000.0
#
# # 预测与真实语音端点
# predicted_segments = [0.475, 5.6, 6.2, 12.575, 13.05, 15.275]
# true_segments = [0.44, 2.02, 2.05, 5.67, 6.14, 10.05, 10.47, 11.58, 11.67, 12.59, 13.43, 15.13]
#
# def time_to_frame(time_list, frame_length):
#     return [int(t / frame_length) for t in time_list]
#
# def generate_binary_labels(segments, total_frames):
#     labels = np.zeros(total_frames)
#     for i in range(0, len(segments), 2):
#         start_frame = time_to_frame([segments[i]], frame_length_sec)[0]
#         end_frame = time_to_frame([segments[i + 1]], frame_length_sec)[0]
#         if start_frame < end_frame:
#             labels[start_frame:end_frame] = 1
#     return labels
#
# max_time = max(max(predicted_segments), max(true_segments))
# total_frames = int(np.ceil(max_time / frame_length_sec))
#
# predicted_labels = generate_binary_labels(predicted_segments, total_frames)
# true_labels = generate_binary_labels(true_segments, total_frames)
#
# # 计算Accuracy
# accuracy = accuracy_score(true_labels, predicted_labels)
# print(f"Accuracy: {accuracy}")
#
# # ROC曲线和AUC值
# fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)
# auc_value = auc(fpr, tpr)
# print(f"AUC: {auc_value}")
#
# # EER计算
# fnr = 1 - tpr
# idx = np.nanargmin(np.absolute((fnr - fpr)))
# eer = (fpr[idx] + fnr[idx]) / 2
# print(f"EER: {eer}")
#
# # 绘制ROC曲线
# plt.figure()
# plt.plot(fpr, tpr, label=f'AUC = {auc_value:.2f}')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()
