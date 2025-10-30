import os
import argparse
import glob
from typing import List, Tuple

import cv2
import numpy as np
import torch
import pandas as pd

from config_enhanced import config
from models_enhanced import create_simplified_model, SimplifiedTwoStepModel
from dataset_enhanced import get_enhanced_transforms


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def list_videos(input_path: str) -> List[str]:
    """Return a de-duplicated list of video file paths from a file or a directory."""
    if os.path.isdir(input_path):
        paths = []
        for ext in VIDEO_EXTS:
            paths.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
        # De-duplicate with case-insensitive normalization (Windows)
        seen = set()
        unique_paths = []
        for p in sorted(paths):
            key = os.path.normcase(os.path.abspath(p))
            if key not in seen:
                seen.add(key)
                unique_paths.append(p)
        return unique_paths
    if os.path.isfile(input_path) and input_path.lower().endswith(VIDEO_EXTS):
        return [input_path]
    raise FileNotFoundError(f"No video(s) found at: {input_path}")


def sample_frame_indices(total_frames: int, num_frames: int, stride: int) -> List[int]:
    """Uniformly sample frame indices similar to dataset logic."""
    if total_frames <= 0:
        return list(range(num_frames))
    if total_frames <= num_frames:
        idx = list(range(total_frames)) * (num_frames // total_frames + 1)
        return idx[:num_frames]
    step = max(1, total_frames // (num_frames * stride))
    idx = list(range(0, total_frames, step))[:num_frames]
    while len(idx) < num_frames:
        idx.append(idx[-1])
    return idx


def extract_frames(video_path: str, num_frames: int, stride: int) -> np.ndarray:
    """Read frames from a video without saving to disk; returns [T, H, W, C] uint8."""
    if not os.path.exists(video_path):
        return np.zeros((num_frames, 128, 128, 3), dtype=np.uint8)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return np.zeros((num_frames, 128, 128, 3), dtype=np.uint8)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(total_frames, num_frames, stride)
    frames = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((128, 128, 3), dtype=np.uint8))
    cap.release()
    arr = np.stack(frames, axis=0)
    return arr


def preprocess_frames(frames_bgr: np.ndarray) -> torch.Tensor:
    """Apply validation transforms; returns tensor [T, C, H, W] float32."""
    transform = get_enhanced_transforms('val')
    processed = []
    for f in frames_bgr:
        # Albumentations expects RGB
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        out = transform(image=rgb)["image"]
        processed.append(out)
    tensor = torch.stack(processed, dim=0)  # [T, C, H, W]
    return tensor.float()


@torch.no_grad()
def predict_video(model: torch.nn.Module, device: torch.device, video_path: str) -> Tuple[int, str, float, int, str, float]:
    """Run model on a single video and return predictions and confidences.

    Returns: (action_idx, action_name, action_conf, actor_idx, actor_name, actor_conf)
    """
    frames = extract_frames(video_path, config.num_frames, config.temporal_stride)
    frames_t = preprocess_frames(frames).unsqueeze(0).to(device)  # [1, T, C, H, W]

    outputs = model(frames_t)
    action_probs = outputs.get('action_probs')  # [1, num_actions]
    actor_probs = outputs.get('actor_probs')    # [1, num_actors]

    action_idx = int(torch.argmax(action_probs, dim=1).item())
    actor_idx = int(torch.argmax(actor_probs, dim=1).item())

    action_conf = float(action_probs[0, action_idx].item())
    actor_conf = float(actor_probs[0, actor_idx].item())

    action_name = config.action_classes[action_idx]
    actor_name = config.actor_classes[actor_idx]

    return action_idx, action_name, action_conf, actor_idx, actor_name, actor_conf


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Create model and load weights from checkpoint, adapting hidden_dim if needed."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('model_state_dict', ckpt)

    # Infer hidden_dim from checkpoint to avoid size mismatch
    if 'feature_projection.weight' in state:
        detected_hidden = state['feature_projection.weight'].shape[0]
    elif 'action_classifier.0.weight' in state:
        detected_hidden = state['action_classifier.0.weight'].shape[1]
    else:
        detected_hidden = config.hidden_dim

    model = SimplifiedTwoStepModel(
        backbone=config.backbone,
        num_action_classes=len(config.action_classes),
        num_actor_classes=len(config.actor_classes),
        shared_backbone=config.shared_backbone,
        dropout_rate=config.dropout_rate,
        hidden_dim=detected_hidden,
    )
    model.to(device)
    model.eval()

    model.load_state_dict(state)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced two-step inference on video(s)")
    parser.add_argument("--input", required=True, help="Path to a video file or a folder of videos")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth with model_state_dict")
    parser.add_argument("--device", default=config.device, help="cuda or cpu (default: from config)")
    parser.add_argument("--save_csv", default=None, help="Optional path to save predictions CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    video_paths = list_videos(args.input)
    if not video_paths:
        raise FileNotFoundError("No input videos found.")

    model = load_model(args.checkpoint, device)

    rows = []
    for vp in video_paths:
        a_idx, a_name, a_conf, r_idx, r_name, r_conf = predict_video(model, device, vp)
        print(f"{os.path.basename(vp)} -> action={a_name} ({a_conf:.3f}), actor={r_name} ({r_conf:.3f})")
        rows.append({
            "video_path": vp,
            "action_idx": a_idx,
            "action_name": a_name,
            "action_confidence": a_conf,
            "actor_idx": r_idx,
            "actor_name": r_name,
            "actor_confidence": r_conf,
        })

    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(args.save_csv, index=False)
        print(f"Saved predictions to {args.save_csv}")


if __name__ == "__main__":
    main()


