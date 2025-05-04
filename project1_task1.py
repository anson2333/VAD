import os
import subprocess
import numpy as np
import librosa
from sklearn.metrics import classification_report
import librosa.display
from scipy.signal import lfilter, get_window
from scipy.stats import skew, kurtosis
import soundfile as sf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
from evaluate import *

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 当坐标轴有负号的时候可以显示负号


def _print(bl=True, s=None):
    if bl:
        print(s)
    else:
        pass



class RhythmFeatures:
    """韵律学特征"""
    def __init__(self, input_file, sr=None, frame_len=512, n_fft=None, win_step=2 / 3, window="hamming"):
        """
        初始化
        :param input_file: 输入音频文件
        :param sr: 所输入音频文件的采样率，默认为None
        :param frame_len: 帧长，默认512个采样点(32ms,16kHz),与窗长相同
        :param n_fft: FFT窗口的长度，默认与窗长相同
        :param win_step: 窗移，默认移动2/3，512*2/3=341个采样点(21ms,16kHz)
        :param window: 窗类型，默认汉明窗
        """
        self.input_file = input_file
        self.frame_len = frame_len  # 帧长，单位采样点数
        self.wave_data, self.sr = librosa.load(self.input_file, sr=sr)
        self.window_len = frame_len  # 窗长512
        if n_fft is None:
            self.fft_num = self.window_len  # 设置NFFT点数与窗长相等
        else:
            self.fft_num = n_fft
        self.win_step = win_step
        self.hop_length = round(self.window_len * win_step)  # 重叠部分采样点数设置为窗长的1/3（1/3~1/2）,即帧移(窗移)2/3
        self.window = window

    def lld(self, **kwargs):
        """
        LLDs（low level descriptors）指的是手工设计的一些低水平特征。
        LLDs一般是在一帧frame语音上进行的计算，是用来表示一帧语音的特征。
        :param kwargs: activity_detect参数
        :return: 浊音(1，n)、轻音段(1，2*n)、有效语音段持续时间(1，n)，单位ms,numpy.uint32
                基频F0，单位Hz、一阶、二阶差分(1，按列拉直提取非0元素后个数，>=n_frames),numpy.float32
                对数能量值、一阶、二阶差分(1，n_frames),numpy.float32
                短时能量、一阶、二阶差分(1，无加窗n_frames),numpy.float64
                过零率，单位次,uint32、一阶、二阶差分(1，无加窗n_frames),numpy.float64
                声压级，单位dB、一阶、二阶差分(1，无加窗n_frames),numpy.float64
        """
        duration_voiced, duration_unvoiced, duration_all = self.duration(**kwargs)
        f0, mag = self.pitch()
        f0 = f0.T[np.nonzero(f0.T)]  # 按列提取非0元素，组成一维数组
        f0_de = librosa.feature.delta(f0, width=3)
        f0_de2 = librosa.feature.delta(f0, width=3, order=2)
        energy = np.log(self.energy())
        energy_de = librosa.feature.delta(energy, width=3)
        energy_de2 = librosa.feature.delta(energy, width=3, order=2)
        ste = self.short_time_energy()
        ste_de = librosa.feature.delta(ste, width=3)
        ste_de2 = librosa.feature.delta(ste, width=3, order=2)
        zcr = self.zero_crossing_rate()
        zcr_de = librosa.feature.delta(zcr, width=3)
        zcr_de2 = librosa.feature.delta(zcr, width=3, order=2)
        spl = self.intensity()
        spl_de = librosa.feature.delta(spl, width=3)
        spl_de2 = librosa.feature.delta(spl, width=3, order=2)
        return (duration_voiced, duration_unvoiced, duration_all, f0, f0_de, f0_de2, energy, energy_de, energy_de2,
                ste, ste_de, ste_de2, zcr, zcr_de, zcr_de2, spl, spl_de, spl_de2)

    def hsf(self, **kwargs):
        """
        HSFs（high level statistics functions）是在LLDs的基础上做一些统计而得到的特征，比如均值，最值等。
        HSFs是对一段语音utterance上的多帧语音做统计，是用来表示一个utterance的特征。
        :param kwargs: lld参数:activity_detect参数
        :return: 1*120维HSFs特征,numpy.float64: 浊音/轻音/有效语音段duration的最小值/最大值/极差/均值/标准差（第0-14维）；
                 F0/F0_de/F0_de2的最小值/最大值/极差/均值/标准差/偏度/峰度（第15-35维）；
                 energy/energy_de/energy_de2的最小值/最大值/极差/均值/标准差/偏度/峰度（第36-56维）；
                 ste/ste_de/ste_de2的最小值/最大值/极差/均值/标准差/偏度/峰度（第57-77维）；
                 zcr/zcr_de/zcr_de2的最小值/最大值/极差/均值/标准差/偏度/峰度（第78-98维）；
                 spl/spl_de/spl_de2的最小值/最大值/极差/均值/标准差/偏度/峰度（第99-119维）
        """
        llds = self.lld(**kwargs)
        hsfs = []
        for i in range(len(llds)):
            hsfs = np.append(hsfs, [np.min(llds[i]), np.max(llds[i]),
                                    np.ptp(llds[i]), np.mean(llds[i]), np.std(llds[i])])
            if i > 2:  # 前3个为duration，不计算其偏度和峰度
                hsfs = np.append(hsfs, [skew(llds[i]), kurtosis(llds[i])])
        return hsfs

    def short_time_energy(self):
        """
        计算语音短时能量：每一帧中所有语音信号的平方和
        :return: 语音短时能量列表(值范围0-每帧归一化后能量平方和，这里帧长512，则最大值为512)，
        np.ndarray[shape=(1，无加窗，帧移为0的n_frames), dtype=float64]
        """
        energy = []  # 语音短时能量列表
        energy_sum_per_frame = 0  # 每一帧短时能量累加和
        for i in range(len(self.wave_data)):  # 遍历每一个采样点数据
            energy_sum_per_frame += self.wave_data[i] ** 2  # 求语音信号能量的平方和
            if (i + 1) % self.frame_len == 0:  # 一帧所有采样点遍历结束
                energy.append(energy_sum_per_frame)  # 加入短时能量列表
                energy_sum_per_frame = 0  # 清空和
            elif i == len(self.wave_data) - 1:  # 不满一帧，最后一个采样点
                energy.append(energy_sum_per_frame)  # 将最后一帧短时能量加入列表
        energy = np.array(energy)
        energy = np.where(energy == 0, np.finfo(np.float64).eps, energy)  # 避免能量值为0，防止后续取log出错(eps是取非负的最小值)
        return energy

    def zero_crossing_rate(self):
        """
        计算语音短时过零率：单位时间(每帧)穿过横轴（过零）的次数
        :return: 每帧过零率次数列表，np.ndarray[shape=(1，无加窗，帧移为0的n_frames), dtype=uint32]
        """
        zcr = []  # 语音短时过零率列表
        counting_sum_per_frame = 0  # 每一帧过零次数累加和，即过零率
        for i in range(len(self.wave_data)):  # 遍历每一个采样点数据
            if i % self.frame_len == 0:  # 开头采样点无过零，因此每一帧的第一个采样点跳过
                continue
            if self.wave_data[i] * self.wave_data[i - 1] < 0:  # 相邻两个采样点乘积小于0，则说明穿过横轴
                counting_sum_per_frame += 1  # 过零次数加一
            if (i + 1) % self.frame_len == 0:  # 一帧所有采样点遍历结束
                zcr.append(counting_sum_per_frame)  # 加入短时过零率列表
                counting_sum_per_frame = 0  # 清空和
            elif i == len(self.wave_data) - 1:  # 不满一帧，最后一个采样点
                zcr.append(counting_sum_per_frame)  # 将最后一帧短时过零率加入列表
        return np.array(zcr, dtype=np.uint32)

    def energy(self):
        """
        每帧内所有采样点的幅值平方和作为能量值
        :return: 每帧能量值，np.ndarray[shape=(1，n_frames), dtype=float64]
        """
        mag_spec = np.abs(librosa.stft(self.wave_data, n_fft=self.fft_num, hop_length=self.hop_length,
                                       win_length=self.frame_len, window=self.window))
        pow_spec = np.square(mag_spec)
        energy = np.sum(pow_spec, axis=0)
        energy = np.where(energy == 0, np.finfo(np.float64).eps, energy)  # 避免能量值为0，防止后续取log出错(eps是取非负的最小值)
        return energy

    def intensity(self):
        """
        计算声音强度，用声压级表示：每帧语音在空气中的声压级Sound Pressure Level(SPL)，单位dB
        公式：20*lg(P/Pref)，P为声压（Pa），Pref为参考压力(听力阈值压力)，一般为2.0*10-5 Pa
        这里P认定为声音的幅值：求得每帧所有幅值平方和均值，除以Pref平方，再取10倍lg
        :return: 每帧声压级，dB，np.ndarray[shape=(1，无加窗，帧移为0的n_frames), dtype=float64]
        """
        p0 = 2.0e-5  # 听觉阈限压力auditory threshold pressure: 2.0*10-5 Pa
        e = self.short_time_energy()
        spl = 10 * np.log10(1 / (np.power(p0, 2) * self.frame_len) * e)
        return spl

    def duration(self, **kwargs):
        """
        持续时间：浊音、轻音段持续时间，有效语音段持续时间,一段有效语音段由浊音段+浊音段两边的轻音段组成
        :param kwargs: activity_detect参数
        :return: np.ndarray[dtype=uint32],浊音shape=(1，n)、轻音段shape=(1，2*n)、有效语音段持续时间列表shape=(1，n)，单位ms
        """
        wav_dat_split_f, wav_dat_split, voiced_f, unvoiced_f = self.activity_detect(**kwargs)  # 端点检测
        duration_voiced = []  # 浊音段持续时间
        duration_unvoiced = []  # 轻音段持续时间
        duration_all = []  # 有效语音段持续时间
        if np.array(voiced_f).size > 1:  # 避免语音过短，只有一帧浊音段
            for voiced in voiced_f:  # 根据帧分割计算浊音段持续时间，两端闭区间
                duration_voiced.append(round((voiced[1] - voiced[0] + 1) * self.frame_len / self.sr * 1000))
        else:  # 只有一帧时
            duration_voiced.append(round(self.frame_len / self.sr * 1000))
        for unvoiced in unvoiced_f:  # 根据帧分割计算清音段持续时间，浊音段左侧左闭右开，浊音段右侧左开右闭
            duration_unvoiced.append(round((unvoiced[1] - unvoiced[0]) * self.frame_len / self.sr * 1000))
        if len(duration_unvoiced) <= 1:  # 避免语音过短，只有一帧浊音段
            duration_unvoiced.append(0)
        for i in range(len(duration_voiced)):  # 浊音段+浊音段两边的轻音段组成一段有效语音段
            duration_all.append(duration_unvoiced[i * 2] + duration_voiced[i] + duration_unvoiced[i * 2 + 1])
        return (np.array(duration_voiced, dtype=np.uint32), np.array(duration_unvoiced, dtype=np.uint32),
                np.array(duration_all, dtype=np.uint32))

    def pitch(self, ts_mag=0.25):
        """
        获取每帧音高，即基频，这里应该包括基频和各次谐波，最小的为基频（一次谐波），其他的依次为二次、三次...谐波
        各次谐波等于基频的对应倍数，因此基频也等于各次谐波除以对应的次数，精确些等于所有谐波之和除以谐波次数之和
        :param ts_mag: 幅值倍乘因子阈值，>0，大于np.average(np.nonzero(magnitudes)) * ts_mag则认为对应的音高有效,默认0.25
        :return: 每帧基频及其对应峰的幅值(>0)，
                 np.ndarray[shape=(1 + n_fft/2，n_frames), dtype=float32]，（257，全部采样点数/(512*2/3)+1）
        """
        mag_spec = np.abs(librosa.stft(self.wave_data, n_fft=self.fft_num, hop_length=self.hop_length,
                                       win_length=self.frame_len, window=self.window))
        # pitches:shape=(d,t)  magnitudes:shape=(d.t), Where d is the subset of FFT bins within fmin and fmax.
        # pitches[f,t] contains instantaneous frequency at bin f, time t
        # magnitudes[f,t] contains the corresponding magnitudes.
        # pitches和magnitudes大于maximal magnitude时认为是一个pitch，否则取0，maximal默认取threshold*ref(S)=1*mean(S, axis=0)
        pitches, magnitudes = librosa.piptrack(S=mag_spec, sr=self.sr, threshold=1.0, ref=np.mean,
                                               fmin=50, fmax=500)  # 人类正常说话基频最大可能范围50-500Hz
        ts = np.average(magnitudes[np.nonzero(magnitudes)]) * ts_mag
        pit_likely = pitches
        mag_likely = magnitudes
        pit_likely[magnitudes < ts] = 0
        mag_likely[magnitudes < ts] = 0
        return pit_likely, mag_likely

    def activity_detect(self, min_interval=15, e_low_multifactor=1.0, zcr_multifactor=1.0, pt=False):
        """
        利用短时能量，短时过零率，使用双门限法进行端点检测
        :param min_interval: 最小浊音间隔，默认15帧
        :param e_low_multifactor: 能量低阈值倍乘因子，默认1.0
        :param zcr_multifactor: 过零率阈值倍乘因子，默认1.0
        :param pt: 输出打印标志位，默认为False
        :return: 全部有效语音段:按帧分割后(list,n*2)、按全部采样点的幅值分割(np.ndarray[shape=(n, 采样值数), dtype=float32])、
                浊音段(list,n*2)、轻音段(list,n*2)
        """
        ste = self.short_time_energy()
        zcr = self.zero_crossing_rate()
        energy_average = sum(ste) / len(ste)  # 求全部帧的短时能量均值
        energy_high = energy_average / 4  # 能量均值的4分之一作为能量高阈值
        energy_low = (sum(ste[:5]) / 5 + energy_high / 5) * e_low_multifactor  # 前5帧能量均值+能量高阈值的5分之一作为能量低阈值
        zcr_threshold = sum(zcr) / len(zcr) * zcr_multifactor  # 过零率均值*zcr_multfactor作为过零率阈值
        voiced_sound = []  # 语音段的浊音部分
        voiced_sound_added = []  # 浊音扩充后列表
        wave_detected = []  # 轻音扩充后的最终列表
        # 首先利用能量高阈值energy_high进行初步检测，得到语音段的浊音部分
        add_flag = True  # 加入voiced_sound列表标志位
        for i in range(len(ste)):  # 遍历短时能量数据
            if len(voiced_sound) == 0 and add_flag and ste[i] >= energy_high:  # 第一次达到阈值
                voiced_sound.append(i)  # 加入列表
                add_flag = False  # 接下来禁止加入
            if (not add_flag) and ste[i] < energy_high:  # 直到未达到阈值，此时该阶段为一段浊音语音
                if i - voiced_sound[-1] <= 2:  # 检测帧索引间隔，去掉间隔小于2的索引，判断该段为噪音
                    voiced_sound = voiced_sound[:-1]  # 该段不加入列表
                else:  # 否则加入列表
                    voiced_sound.append(i)
                add_flag = True  # 继续寻找下一段浊音（下一个阈值）
            # 再次达到阈值，判断两个浊音间隔是否大于最小浊音间隔
            elif add_flag and ste[i] >= energy_high and i - voiced_sound[-1] > min_interval:
                voiced_sound.append(i)  # 大于，则分段，加入列表
                add_flag = False  # 接下来禁止加入
            elif add_flag and ste[i] >= energy_high and i - voiced_sound[-1] <= min_interval:
                voiced_sound = voiced_sound[:-1]  # 小于，则不分段，该段不加入列表
                add_flag = False  # 接下来禁止加入
            if (i == len(ste) - 1) and (len(voiced_sound) % 2 == 1):  # 当到达最后一帧，发现浊音段为奇数，则此时到最后一帧为浊音段
                if i - voiced_sound[-1] <= 2:  # 检测帧索引间隔，去掉间隔小于2的索引，判断该段为噪音
                    voiced_sound = voiced_sound[:-1]  # 该段不加入列表
                else:  # 否则加入列表
                    voiced_sound.append(i)
        _print(pt, "能量高阈值:{}，浊音段:{}".format(energy_high, voiced_sound))
        # 再通过能量低阈值energy_low在浊音段向两端进行搜索，超过energy_low便视为有效语音
        for j in range(len(voiced_sound)):  # 遍历浊音列表
            i_minus_flag = False  # i值减一标志位
            i = voiced_sound[j]  # 浊音部分帧索引
            if j % 2 == 1:  # 每段浊音部分的右边帧索引
                while i < len(ste) and ste[i] >= energy_low:  # 搜索超过能量低阈值的帧索引
                    i += 1  # 向右搜索
                voiced_sound_added.append(i)  # 搜索到则加入扩充列表，右闭
            else:  # 每段浊音部分的左边帧索引
                while i > 0 and ste[i] >= energy_low:  # 搜索超过能量低阈值的帧索引
                    i -= 1  # 向左搜索
                    i_minus_flag = True  # i值减一标志位置位
                if i_minus_flag:  # 搜索到则加入扩充列表，左闭
                    voiced_sound_added.append(i + 1)
                else:
                    voiced_sound_added.append(i)
        _print(pt, "能量低阈值:{}，浊音再次扩展后:{}".format(energy_low, voiced_sound_added))
        # 最后通过过零率对浊音扩充后列表向两端再次进行搜索，获取轻音部分
        for j in range(len(voiced_sound_added)):  # 遍历浊音扩充后列表
            i_minus_flag = False  # i值减一标志位
            i = voiced_sound_added[j]  # 浊音扩充后部分帧索引
            if j % 2 == 1:  # 每段浊音扩充部分的右边帧索引
                while i < len(zcr) and zcr[i] >= zcr_threshold:  # 搜索超过过零率阈值的帧索引
                    i += 1  # 向右搜索
                wave_detected.append(i)  # 搜索到则加入扩充列表，右开
            else:  # 每段浊音扩充部分的左边帧索引
                while i > 0 and zcr[i] >= zcr_threshold:  # 搜索超过过零率阈值的帧索引
                    i -= 1  # 向左搜索
                    i_minus_flag = True  # i值减一标志位置位
                if i_minus_flag:  # 搜索到则加入扩充列表，左闭
                    wave_detected.append(i + 1)
                else:
                    wave_detected.append(i)
        _print(pt, "过零率阈值:{}，轻音段增加后:{}".format(zcr_threshold, wave_detected))
        wave_data_detected_frame = []  # 端点检测后，以帧为单位的有效语音列表
        for index in range(len(wave_detected)):
            if index % 2 == 0:  # 按段分割成列表
                wave_data_detected_frame.append(wave_detected[index:index + 2])
            else:
                continue
        _print(pt, "分割后共{}段语音，按帧分割为{}".format(len(wave_data_detected_frame), wave_data_detected_frame))
        wave_data_detected = []  # 端点检测后，对应全部采样点的幅值列表，其中列表代表每个有效语音段
        for index in wave_data_detected_frame:
            try:
                wave_data_detected.append(self.wave_data[index[0] * int(self.frame_len):
                                                         index[1] * int(self.frame_len)])
            except IndexError:
                wave_data_detected.append(self.wave_data[index[0] * int(self.frame_len):-1])
        _print(pt, "分割后共{}段语音，按全部采样点的幅值分割为{}".format(len(wave_data_detected), wave_data_detected))
        if np.array(voiced_sound_added).size > 1:  # 避免语音过短，只有一帧浊音段
            voiced_frame = np.array(voiced_sound_added).reshape((-1, 2)).tolist()  # 按帧分割的浊音段
        else:  # 只有一帧时
            voiced_frame = np.array(voiced_sound_added).tolist()
        unvoiced_frame = []  # 按帧分割的轻音段
        for i in range(len(wave_detected)):  # 根据最终的扩充后列表和浊音段列表求得轻音段
            if wave_detected[i] < voiced_sound_added[i]:
                unvoiced_frame.append([wave_detected[i], voiced_sound_added[i]])
            elif wave_detected[i] > voiced_sound_added[i]:
                unvoiced_frame.append([voiced_sound_added[i], wave_detected[i]])
            else:
                unvoiced_frame.append([0, 0])
        return wave_data_detected_frame, wave_data_detected, voiced_frame, unvoiced_frame

class VAD:
    """语音端点检测"""
    def __init__(self, wav_file, frame_len=400, min_interval=15, e_low_multifactor=1.0, zcr_multifactor=1.0, pt=True):
        """
        初始化函数
        语音信号是非平稳信号，但是可以认为10~30ms的时间范围内，语音信号是平稳信号,比如这里我取25ms作为一帧
        此时一帧包含25ms*采样率(16kHz)*通道数（1）=400个采样点
        :param wav_file: 输入.wav音频文件
        :param frame_len: 帧长，默认400个采样点
        :param min_interval: 最小浊音间隔，默认15帧
        :param e_low_multifactor: 能量低阈值倍乘因子，默认1.0
        :param zcr_multifactor: 过零率阈值倍乘因子，默认1.0
        :param pt: 输出打印标志位，默认为True
        """
        rf = RhythmFeatures(wav_file, None, frame_len)
        self.wave_data = rf.wave_data  # 获取音频全部采样点的数组形式数据,每个采样点类型为np.float32
        self.sampling_rate = rf.sr
        self.frame_len_samples = frame_len  # 帧长，单位采样点数
        self.frame_len_time = round(self.frame_len_samples * 1000 / self.sampling_rate)  # 帧长，单位ms
        self.energy = rf.short_time_energy()  # 获取短时能量
        self.zcr = rf.zero_crossing_rate()  # 获取过零率
        # 获取端点检测后的有效语音段
        self.wav_dat_split_f, self.wav_dat_split, self.voiced_f, self.unvoiced_f = \
            rf.activity_detect(min_interval, e_low_multifactor, zcr_multifactor, pt)
        # 语音首尾端点检测，中间不检测
        if len(self.wav_dat_split_f[-1]) > 1:  # 避免语音过短，只有一帧
            self.wav_dat_utterance = self.wave_data[self.wav_dat_split_f[0][0] * int(self.frame_len_samples):
                                                    self.wav_dat_split_f[-1][1] * int(self.frame_len_samples)]
        else:  # 只有一帧时
            self.wav_dat_utterance = self.wave_data[self.wav_dat_split_f[0][0] * int(self.frame_len_samples):]

    def return_wav_dat_split_f(self):
        return self.wav_dat_split_f

    def return_wav_dat_split(self):
        wav = []
        for i in range(len(self.wav_dat_split_f)):  # 端点检测分割线
            for j in range(len(self.wav_dat_split_f[i])):
                wav.append(self.wav_dat_split_f[i][j] * self.frame_len_time / 1000)
        return wav

    def plot(self):
        """
        绘制音频波形、短时能量和过零率曲线
        :return: None
        """
        audio_total_time = int(len(self.wave_data) / self.sampling_rate * 1000)  # 音频总时间
        plt.figure(figsize=(16, 6))
        # 以下绘制短时能量曲线
        plt.subplot(1, 3, 2)
        frames = [i for i in range(0, len(self.energy))]  # 横轴为帧数轴
        plt.title("Short Time Energy")
        plt.xlabel("Frames")
        plt.ylabel("Energy")
        plt.plot(frames, self.energy, c="g", lw=1)
        plt.grid()
        # 以下绘制过零率曲线
        plt.subplot(1, 3, 3)
        frames = [i for i in range(0, len(self.zcr))]  # 横轴为帧数轴
        plt.title("Zero Crossing Rate")
        plt.xlabel("Frames")
        plt.ylabel("Times of Zero Crossing")
        plt.plot(frames, self.zcr, c="r", lw=1)
        plt.grid()
        # 以下绘制语音波形曲线+端点检测分割线
        plt.subplot(1, 3, 1)
        plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
        time = [int(i * (audio_total_time / len(self.wave_data))) for i in range(0, len(self.wave_data))]
        plt.title("Wave Form")
        plt.xlabel("Time/ms")
        plt.ylabel("Normalized Amplitude")
        plt.ylim(-1, 1)
        plt.plot(time, self.wave_data, c="b", lw=1)  # 语音波形曲线
        c = "g"

        for i in range(len(self.wav_dat_split_f)):  # 端点检测分割线
            for j in range(len(self.wav_dat_split_f[i])):
                if (i == 0 and j == 0) or (i == len(self.wav_dat_split_f) - 1 and j == 1):
                    plt.axvline(x=self.wav_dat_split_f[i][j] * self.frame_len_time, c=c, ls="-", lw=2)
                else:
                    plt.axvline(x=self.wav_dat_split_f[i][j] * self.frame_len_time, c=c, ls="--", lw=1.5)
            if c == "r":
                c = "g"
            else:
                c = "r"
        plt.grid()

        plt.tight_layout()
        plt.show()

def build_audio_segments_dict(file_path):
    """
    将标注文件加载为字典格式 {audio_id: segments}
    :param file_path: 标注文件路径
    :return: 字典（如 {"1031-133220-0062": [(0.44,2.02), ...], ...}）
    """
    segments_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                audio_id = parts[0]
                # 将所有时间点平铺存储（不分组）
                segments = []
                for seg in parts[1:]:
                    segments.extend(map(float, seg.split(',')))
                segments_dict[audio_id] = segments
    return segments_dict

def process_audio_files(input_dir, output_file):
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
                wav_path = os.path.join(input_dir, filename)
                new_filename = filename.replace(".wav","")
                # 进行端点检测
                vad = VAD(wav_path, min_interval=15, pt=False)
                segments = vad.return_wav_dat_split()

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


if __name__ == "__main__":
    # 输入输出路径设置
    input_dir = r"voice-activity-detection-sjtu-spring-2024\vad\wavs\test"
    output_file = "vad_test_results_task1.txt"

    # 确保输入目录存在
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Directory not found: {input_dir}")

    # 处理所有音频文件
    process_audio_files(input_dir, output_file)


    print(f"处理完成，结果已保存到 {output_file}")

# if __name__ == "__main__":
#     current_path = os.getcwd()
#     print(current_path)
#
#     wave_file = os.path.join(current_path, "14-208-0048.wav")
#     wave_file_vad = wave_file.split(".")[0] + "_vad.wav"
#     # feature_file = os.path.join(current_path, "features/feature.csv")
#     # np.set_printoptions(threshold=np.inf)
#     # 端点检测
#     vad = VAD(wave_file, min_interval=15, pt=False)
#     vad.plot()
#     print(vad.return_wav_dat_split())
#     # # 利用openSmile工具进行特征提取
#     # opensmile_f = OpenSmileFeatureSet(wave_file_vad)
#     # feat = opensmile_f.get_IS09(feature_file)
#     # print(feat.shape, feat.dtype, feat)
#     # # 常用声学特征
#     # my_acoustic_f = my_acoustic_features(wave_file_vad)
#     # print(my_acoustic_f.shape, my_acoustic_f.dtype, my_acoustic_f)
#     # # 韵律学特征
#     # rhythm_f = RhythmFeatures(wave_file_vad)
#     # rhythm_f.plot()
#     # # 基于谱的相关特征
#     # spectrum_f = SpectrumFeatures(wave_file_vad)
#     # spectrum_f.plot()
#     # # 音质特征
#     # quality_f = QualityFeatures(wave_file_vad)
#     # quality_f.plot()
#     # # 声谱图特征
#     # spectrogram_f = Spectrogram(wave_file_vad)
#     # spectrogram_f.plot()
