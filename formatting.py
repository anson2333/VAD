# # # def build_audio_segments_dict(file_path):
# # #     """
# # #     将标注文件加载为字典格式 {audio_id: segments}
# # #     :param file_path: 标注文件路径
# # #     :return: 字典（如 {"1031-133220-0062": [(0.44,2.02), ...], ...}）
# # #     """
# # #     segments_dict = {}
# # #     with open(file_path, 'r', encoding='utf-8') as f:
# # #         for line in f:
# # #             parts = line.strip().split()
# # #             if parts:
# # #                 audio_id = parts[0]
# # #                 # 将所有时间点平铺存储（不分组）
# # #                 segments = []
# # #                 for seg in parts[1:]:
# # #                     segments.extend(map(float, seg.split(',')))
# # #                 segments_dict[audio_id] = segments
# # #     return segments_dict
# # #
# # # # 使用示例
# # # segments_dict = build_audio_segments_dict("voice-activity-detection-sjtu-spring-2024\\vad\data\dev_label.txt")
# # # target_id = "1031-133220-0062"
# # # if target_id in segments_dict:
# # #     print(f"{target_id} 的语音端点：{segments_dict[target_id]}")
# # # else:
# # #     print("未找到指定音频ID")
# #
# # import wave
# #
# # def get_wav_sample_rate(file_path):
# #     with wave.open(file_path, 'rb') as wav_file:
# #         sample_rate = wav_file.getframerate()
# #         return sample_rate
# #
# # # 示例：替换为你的音频文件路径
# # audio_file_path = "14-208-0048.wav"
# # sample_rate = get_wav_sample_rate(audio_file_path)
# # print(f"采样率: {sample_rate} Hz")
#
# # with open('vad_test_results.txt', 'r') as f_in, open('vad_test_results_task1.txt', 'w') as f_out:
# #     for line in f_in:
# #         # 分割文件名和时间数据
# #         filename, times = line.strip().split('\t')
# #
# #         # 去除.wav后缀
# #         filename = filename.replace('.wav', '')
# #
# #         # 处理时间数据
# #         times = times.strip('[]')  # 去除方括号
# #         times = times.split(', ')  # 分割成列表
# #         times = [t.strip() for t in times]  # 去除可能的空格
# #
# #         # 重新组合时间数据
# #         new_times = f"{times[0]},{times[1]} {times[2]},{times[3]}"
# #
# #         # 写入新文件
# #         f_out.write(f"{filename}\t{new_times}\n")
# with open('vad_test_results.txt', 'r') as f_in, open('vad_test_results_task1.txt', 'w') as f_out:
#     for line in f_in:
#         line = line.strip()
#         if not line:  # 跳过空行
#             continue
#
#         try:
#             # 分割文件名和时间数据
#             filename, times = line.split('\t')
#
#             # 去除.wav后缀
#             filename = filename.replace('.wav', '')
#
#             # 处理时间数据：去除方括号，分割成列表
#             times = times.strip('[]').replace(' ', '')  # 去除所有空格
#             times_list = times.split(',')
#
#             # 检查是否是偶数个时间点（才能两两配对）
#             if len(times_list) % 2 != 0:
#                 print(f"警告：行 '{line}' 的时间数据不是偶数个，无法配对，跳过处理")
#                 continue
#
#             # 生成成对的时间数据，如 "0.55,7.15 8.05,16.175"
#             paired_times = []
#             for i in range(0, len(times_list), 2):
#                 paired_times.append(f"{times_list[i]},{times_list[i + 1]}")
#             new_times = " ".join(paired_times)
#
#             # 写入新文件
#             f_out.write(f"{filename}\t{new_times}\n")
#
#         except Exception as e:
#             print(f"错误：无法处理行 '{line}'，原因：{e}")

# import re
#
# with open('vad_test_results_task2.txt', 'r') as f_in, open('vad_test_results_task2_new.txt', 'w') as f_out:
#     for line in f_in:
#         line = line.strip()
#         if not line:
#             continue
#
#         # 分割文件名和时间数据
#         filename, times = line.split('\t', 1)
#
#         # 去除.wav后缀
#         filename = filename.replace('.wav', '')
#
#         # 使用正则提取所有(x, y)对
#         pairs = re.findall(r'\(([^)]+)\)', times)
#
#         # 处理每个对：去除空格，替换逗号后的空格
#         processed_pairs = []
#         for pair in pairs:
#             cleaned = pair.replace(' ', '').replace(',', ',')  # 确保格式一致
#             processed_pairs.append(cleaned)
#
#         # 组合成目标格式
#         new_times = ' '.join(processed_pairs)
#
#         f_out.write(f"{filename}\t{new_times}\n")
#
# def clean_timestamps_file(input_file, output_file, decimal_places=2):
#     """
#     处理 input.txt，规范化时间戳的浮点数精度
#     :param input_file: 输入文件名（如 "input.txt"）
#     :param output_file: 输出文件名（如 "output.txt"）
#     :param decimal_places: 保留的小数位数（默认 2）
#     """
#     with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
#         for line in f_in:
#             line = line.strip()
#             if not line:
#                 continue  # 跳过空行
#
#             # 分割文件名和时间戳部分
#             parts = line.split('\t')
#             if len(parts) != 2:
#                 f_out.write(line + '\n')  # 如果格式错误，直接写入原行
#                 continue
#
#             filename, timestamps = parts[0], parts[1]
#
#             # 处理时间戳
#             cleaned_pairs = []
#             for pair in timestamps.split():
#                 try:
#                     start, end = pair.split(',')
#                     # 四舍五入到指定小数位
#                     start_clean = round(float(start), decimal_places)
#                     end_clean = round(float(end), decimal_places)
#                     # 格式化为字符串，避免科学计数法（如 1.0 -> "1.0" 而不是 "1"）
#                     start_str = format(start_clean, f".{decimal_places}f")
#                     end_str = format(end_clean, f".{decimal_places}f")
#                     cleaned_pairs.append(f"{start_str},{end_str}")
#                 except ValueError:
#                     cleaned_pairs.append(pair)  # 如果拆分失败，保留原数据
#
#             # 重新组合并写入文件
#             cleaned_timestamps = ' '.join(cleaned_pairs)
#             f_out.write(f"{filename}\t{cleaned_timestamps}\n")
#
#
# # 调用函数处理文件
# clean_timestamps_file("vad_test_results_task2_new.txt", "vad_test_results_task2_new_new.txt", decimal_places=2)

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


# 示例调用
process_file("vad_test_results_task2_new_new.txt", "vad_test_results_task2.txt", min_silence=0.3, min_voice=0.1)
