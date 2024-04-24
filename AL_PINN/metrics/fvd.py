"""
https://github.com/cvpr2022-stylegan-v/stylegan-v/blob/main/src/metrics/frechet_video_distance.py
"""
from typing import Tuple
import scipy
import numpy as np
import cv2
import tensorflow as tf


def compute_fvd(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return float(fid)


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = feats.mean(axis=0)  # [d]
    sigma = np.cov(feats, rowvar=False)  # [d, d]

    return mu, sigma


def load_vid(path):
    cap = cv2.VideoCapture(path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(np.array(frame))
    cap.release()
    video_np = np.array(frames)
    print("Video shape:", video_np.shape)
    return video_np


def extract_features(video_np: np.ndarray, model: tf.keras.Model) -> np.ndarray:
    video_np_resized = tf.image.resize(video_np, (299, 299))  # Resize frames to the input size of Inception-V3
    video_np_normalized = (video_np_resized - 127.5) / 127.5  # Normalize frames to the range of [-1, 1]
    features = model.predict(video_np_normalized)  # [num_frames, num_features]
    return features


if __name__ == "__main__":
    out = load_vid('/data1/lzengaf/dl4cfd/sample_1.mp4')
    gt = load_vid('/data1/lzengaf/dl4cfd/sample_2.mp4')

    inception_model = tf.keras.applications.InceptionV3(include_top=False, pooling="avg", input_shape=(299, 299, 3))
    # Extract features from videos
    out_feats = extract_features(out, inception_model)
    gt_feats = extract_features(gt, inception_model)

    # Calculate FVD
    print(compute_fvd(out_feats, gt_feats))
    # 351.209942027641