import tensorflow as tf
import os
import random
import numpy as np

import librosa
import pyworld


def l1_loss(y, y_hat):

    return tf.reduce_mean(tf.abs(y - y_hat))


def l2_loss(y, y_hat):

    return tf.reduce_mean(tf.square(y - y_hat))


def cross_entropy_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))


def load_wavs(wav_dir, sr):
    wavs = list()
    for file in os.listdir(wav_dir):
        file_path = os.path.join(wav_dir, file)
        wav, _ = librosa.load(file_path, sr=sr, mono=True)
        # wav = wav.astype(np.float64)
        wavs.append(wav)

    return wavs


def world_decompose(wav, fs, frame_period=5.0):
    # Decompose speech signal into f0, spectral envelope and aperiodicity using WORLD
    wav = wav.astype(np.float64)
    f0, timeaxis = pyworld.harvest(wav, fs, frame_period=frame_period, f0_floor=71.0, f0_ceil=800.0)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, fs)
    ap = pyworld.d4c(wav, f0, timeaxis, fs)

    return f0, timeaxis, sp, ap


def world_encode_spectral_envelop(sp, fs, dim=24):
    # Get Mel-cepstral coefficients (MCEPs)

    # sp = sp.astype(np.float64)
    coded_sp = pyworld.code_spectral_envelope(sp, fs, dim)

    return coded_sp


def world_decode_spectral_envelop(coded_sp, fs):
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    # coded_sp = coded_sp.astype(np.float32)
    # coded_sp = np.ascontiguousarray(coded_sp)
    decoded_sp = pyworld.decode_spectral_envelope(coded_sp, fs, fftlen)

    return decoded_sp


def world_encode_data(wavs, fs, frame_period=5.0, coded_dim=24):
    f0s = list()
    timeaxes = list()
    sps = list()
    aps = list()
    coded_sps = list()

    for wav in wavs:
        f0, timeaxis, sp, ap = world_decompose(wav=wav, fs=fs, frame_period=frame_period)
        coded_sp = world_encode_spectral_envelop(sp=sp, fs=fs, dim=coded_dim)
        f0s.append(f0)
        timeaxes.append(timeaxis)
        sps.append(sp)
        aps.append(ap)
        coded_sps.append(coded_sp)

    return f0s, timeaxes, sps, aps, coded_sps


def transpose_in_list(lst):
    transposed_lst = list()
    for array in lst:
        transposed_lst.append(array.T)
    return transposed_lst


def world_decode_data(coded_sps, fs):
    decoded_sps = list()

    for coded_sp in coded_sps:
        decoded_sp = world_decode_spectral_envelop(coded_sp, fs)
        decoded_sps.append(decoded_sp)

    return decoded_sps


def world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period):
    # decoded_sp = decoded_sp.astype(np.float64)
    wav = pyworld.synthesize(f0, decoded_sp, ap, fs, frame_period)
    # Librosa could not save wav if not doing so
    wav = wav.astype(np.float32)

    return wav


def world_synthesis_data(f0s, decoded_sps, aps, fs, frame_period):
    wavs = list()

    for f0, decoded_sp, ap in zip(f0s, decoded_sps, aps):
        wav = world_speech_synthesis(f0, decoded_sp, ap, fs, frame_period)
        wavs.append(wav)

    return wavs


def coded_sps_normalization_fit_transoform(coded_sps):
    coded_sps_concatenated = np.concatenate(coded_sps, axis=1)
    coded_sps_mean = np.mean(coded_sps_concatenated, axis=1, keepdims=True)
    coded_sps_std = np.std(coded_sps_concatenated, axis=1, keepdims=True)

    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)

    return coded_sps_normalized, coded_sps_mean, coded_sps_std


def coded_sps_normalization_transoform(coded_sps, coded_sps_mean, coded_sps_std):
    coded_sps_normalized = list()
    for coded_sp in coded_sps:
        coded_sps_normalized.append((coded_sp - coded_sps_mean) / coded_sps_std)

    return coded_sps_normalized


def coded_sps_normalization_inverse_transoform(normalized_coded_sps, coded_sps_mean, coded_sps_std):
    coded_sps = list()
    for normalized_coded_sp in normalized_coded_sps:
        coded_sps.append(normalized_coded_sp * coded_sps_std + coded_sps_mean)

    return coded_sps


def coded_sp_padding(coded_sp, multiple=4):
    num_features = coded_sp.shape[0]
    num_frames = coded_sp.shape[1]
    num_frames_padded = int(np.ceil(num_frames / multiple)) * multiple
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    coded_sp_padded = np.pad(coded_sp, ((0, 0), (num_pad_left, num_pad_right)), 'constant', constant_values=0)

    return coded_sp_padded


def wav_padding(wav, sr, frame_period, multiple=4):
    assert wav.ndim == 1
    num_frames = len(wav)
    num_frames_padded = int(
        (np.ceil((np.floor(num_frames / (sr * frame_period / 1000)) + 1) / multiple + 1) * multiple - 1) * (
                    sr * frame_period / 1000))
    num_frames_diff = num_frames_padded - num_frames
    num_pad_left = num_frames_diff // 2
    num_pad_right = num_frames_diff - num_pad_left
    wav_padded = np.pad(wav, (num_pad_left, num_pad_right), 'constant', constant_values=0)

    return wav_padded


def logf0_statistics(f0s):
    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = log_f0s_concatenated.std()

    return log_f0s_mean, log_f0s_std


def pitch_conversion(f0, mean_log_src, std_log_src, mean_log_target, std_log_target):
    # Logarithm Gaussian normalization for Pitch Conversions
    try:
        f0_converted = np.exp((np.log(f0) - mean_log_src) / std_log_src * std_log_target + mean_log_target)
    except:
        return f0
    return f0_converted


def wavs_to_specs(wavs, n_fft=1024, hop_length=None):
    stfts = list()
    for wav in wavs:
        stft = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)
        stfts.append(stft)

    return stfts


def wavs_to_mfccs(wavs, sr, n_fft=1024, hop_length=None, n_mels=128, n_mfcc=24):
    mfccs = list()
    for wav in wavs:
        mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, n_mfcc=n_mfcc)
        mfccs.append(mfcc)

    return mfccs


def mfccs_normalization(mfccs):
    mfccs_concatenated = np.concatenate(mfccs, axis=1)
    mfccs_mean = np.mean(mfccs_concatenated, axis=1, keepdims=True)
    mfccs_std = np.std(mfccs_concatenated, axis=1, keepdims=True)

    mfccs_normalized = list()
    for mfcc in mfccs:
        mfccs_normalized.append((mfcc - mfccs_mean) / mfccs_std)

    return mfccs_normalized, mfccs_mean, mfccs_std


# def sample_train_data(pool_A, pool_B, n_frames=128, max_samples=1000):
#     # Sample proportional to the length
#
#     np.random.shuffle(pool_A)
#     np.random.shuffle(pool_B)
#
#     train_data_A = []
#     train_data_B = []
#
#     while pool_A and pool_B:
#
#         idx_A = np.random.randint(len(pool_A))
#         idx_B = np.random.randint(len(pool_B))
#         data_A, data_B = pool_A[idx_A], pool_B[idx_B]
#         data_A_len, data_B_len = data_A.shape[1], data_B.shape[1]
#
#         if data_A_len < n_frames:
#             del pool_A[idx_A]
#             continue
#
#         if data_B_len < n_frames:
#             del pool_B[idx_B]
#             continue
#
#         start_A = np.random.randint(data_A_len - n_frames + 1)
#         end_A = start_A + n_frames
#         train_data_A.append(data_A[:, start_A:end_A])
#         if start_A >= n_frames:
#             pool_A.append(data_A[:, 0:start_A])
#         if data_A_len - end_A >= n_frames:
#             pool_A.append(data_A[:, end_A:])
#         del pool_A[idx_A]
#
#         start_B = np.random.randint(data_B_len - n_frames + 1)
#         end_B = start_B + n_frames
#         train_data_B.append(data_B[:, start_B:end_B])
#         if start_B >= n_frames:
#             pool_B.append(data_B[:, 0:start_B])
#         if data_B_len - end_B >= n_frames:
#             pool_B.append(data_B[:, end_B:])
#         del pool_B[idx_B]
#
#         # reach maximum data length
#         if len(train_data_A) >= max_samples:
#             break
#
#     train_data_A = np.array(train_data_A)
#     train_data_B = np.array(train_data_B)
#
#     return train_data_A, train_data_B


def sample_train_data(pool_A, pool_B, f0s_A, f0s_B, n_frames=128, max_samples=1000):
    # remove silence

    train_data_A = []
    train_data_B = []

    while pool_A and pool_B:

        idx_A = np.random.randint(len(pool_A))
        idx_B = np.random.randint(len(pool_B))
        data_A, data_B = pool_A[idx_A], pool_B[idx_B]
        data_A_len, data_B_len = data_A.shape[1], data_B.shape[1]
        f0_A, f0_B = f0s_A[idx_A], f0s_B[idx_B]

        if data_A_len < n_frames:
            del pool_A[idx_A]
            del f0s_A[idx_A]
            continue

        if data_B_len < n_frames:
            del pool_B[idx_B]
            del f0s_B[idx_B]
            continue

        start_A = np.random.randint(data_A_len - n_frames + 1)
        end_A = start_A + n_frames
        if max(f0_A[start_A:end_A]) > 0:                            # exclude silence waveform
            train_data_A.append(data_A[:, start_A:end_A])
        if start_A >= n_frames and max(f0_A[0:start_A]) > 0:
            pool_A.append(data_A[:, 0:start_A])
            f0s_A.append(f0_A[0:start_A])
        if data_A_len - end_A >= n_frames and max(f0_A[end_A:]) > 0:
            pool_A.append(data_A[:, end_A:])
            f0s_A.append(f0_A[end_A:])
        del pool_A[idx_A]
        del f0s_A[idx_A]

        start_B = np.random.randint(data_B_len - n_frames + 1)
        end_B = start_B + n_frames
        if max(f0_B[start_B:end_B]) > 0:                            # exclude silence waveform
            train_data_B.append(data_B[:, start_B:end_B])
        if start_B >= n_frames and max(f0_B[0:start_B]) > 0:
            pool_B.append(data_B[:, 0:start_B])
            f0s_B.append(f0_B[0:start_B])
        if data_B_len - end_B >= n_frames and max(f0_B[end_B:]) > 0:
            pool_B.append(data_B[:, end_B:])
            f0s_B.append(f0_B[end_B:])
        del pool_B[idx_B]
        del f0s_B[idx_B]

        # reach maximum data length
        if len(train_data_A) >= max_samples:
            break

    num = min(len(train_data_A), len(train_data_B))
    np.random.shuffle(train_data_A)
    np.random.shuffle(train_data_B)
    train_data_A = np.array(train_data_A[0:num])
    train_data_B = np.array(train_data_B[0:num])

    return train_data_A, train_data_B
