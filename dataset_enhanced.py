import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, List, Tuple, Optional
import scipy.io as sio
from PIL import Image
import random
import math
from config_enhanced import config


class EnhancedA2DDataset(Dataset):
    """
    Enhanced A2D Dataset for Two-Step Action Recognition
    Optimized for maximum accuracy with advanced data processing
    """
    
    def __init__(self, 
                 csv_file: str,
                 videos_dir: str,
                 annotations_dir: str,
                 transform=None,
                 mode: str = 'train',
                 num_frames: int = 32,
                 temporal_stride: int = 1,
                 frame_size: Tuple[int, int] = (320, 320)):
        
        self.csv_file = csv_file
        self.videos_dir = videos_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        self.mode = mode
        self.num_frames = num_frames
        self.temporal_stride = temporal_stride
        self.frame_size = frame_size
        
        # Load dataset metadata
        self.data = self._load_metadata()
        self.val_idx_path = f'{config.log_dir}/val_indices.npy'
        self.extract_frames_dir = 'extracted_frames'
        
        # Filter data based on mode
        if mode == 'train':
            train_data = self.data[self.data['usage'] == 0]
            val_size = int(len(self.data) * config.val_split)
            if not os.path.exists(self.val_idx_path):
                val_indices = train_data.sample(n=val_size, random_state=42).index
                os.makedirs(config.log_dir, exist_ok=True)
                np.save(self.val_idx_path, val_indices)
                self.data = train_data.drop(index=val_indices)
            else:
                val_indices = np.load(self.val_idx_path, allow_pickle=True)
                self.data = train_data.drop(index=val_indices)

        elif mode == 'test':
            self.data = self.data[self.data['usage'] == 1]
        elif mode == 'val':
            train_data = self.data[self.data['usage'] == 0]
            if os.path.exists(self.val_idx_path):
                val_indices = np.load(self.val_idx_path, allow_pickle=True)
                self.data = train_data.loc[val_indices]
            else:
                raise FileNotFoundError("Validation indices file not found. Run train mode first.")
        
        # Create label mappings for enhanced two-step approach
        self._create_label_mappings()
        
        print(f"Enhanced A2D Dataset - {mode} mode: {len(self.data)} samples")
        print(f"Action classes: {len(self.action_classes)}")
        print(f"Actor classes: {len(self.actor_classes)}")
    
    def _load_metadata(self) -> pd.DataFrame:
        """Load and process the CSV metadata file with enhanced filtering"""
        df = pd.read_csv(self.csv_file, header=None)
        df.columns = ['video_id', 'label', 'start_time', 'end_time', 
                     'height', 'width', 'num_frames', 'num_annotated_frames', 'usage']
        
        # Filter only valid labels
        df = df[df['label'].isin(config.valid_labels)]
        
        # Extract action and actor from label
        df['action'], df['actor'] = zip(*df['label'].apply(self._decode_label))
        
        # Add additional metadata
        df['duration'] = df['num_frames'] / 30.0  # Approximate duration in seconds
        df['aspect_ratio'] = df['width'] / df['height']
        
        return df
    
    def _decode_label(self, label: int) -> Tuple[str, str]:
        """Decode the two-digit label into action and actor"""
        if label == 0:
            return "background", "background"
        
        # Extract tens and ones digits
        tens = label // 10
        ones = label % 10
        
        # Map to action and actor
        action_map = {
            1: "climbing", 2: "crawling", 3: "eating", 4: "flying",
            5: "jumping", 6: "rolling", 7: "running", 8: "walking", 9: "none"
        }
        
        actor_map = {
            1: "adult", 2: "baby", 3: "ball", 4: "bird",
            5: "car", 6: "cat", 7: "dog"
        }
        
        action = action_map.get(ones, "background")
        actor = actor_map.get(tens, "background")
        
        return action, actor
    
    def _create_label_mappings(self):
        """Create enhanced label mappings for action and actor classification"""
        # Action label mapping
        self.action_classes = config.action_classes
        self.action_to_idx = {action: idx for idx, action in enumerate(self.action_classes)}
        self.idx_to_action = {idx: action for action, idx in self.action_to_idx.items()}
        
        # Actor label mapping
        self.actor_classes = config.actor_classes
        self.actor_to_idx = {actor: idx for idx, actor in enumerate(self.actor_classes)}
        self.idx_to_actor = {idx: actor for actor, idx in self.actor_to_idx.items()}
    
    
    def _extract_frames_enhanced(self, video_path: str, num_frames: int, stride: int, save_dir: str) -> np.ndarray:
        """Extract frames from video with enhanced optimization - no disk saving to avoid errors"""
        try:
            # Check if video exists
            if not os.path.exists(video_path):
                print(f"Warning: Video file not found: {video_path}")
                # Return dummy frames
                dummy_frames = np.zeros((num_frames, 128, 128, 3), dtype=np.uint8)
                return dummy_frames
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open video: {video_path}")
                cap.release()
                # Return dummy frames
                dummy_frames = np.zeros((num_frames, 128, 128, 3), dtype=np.uint8)
                return dummy_frames
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0:
                print(f"Warning: Video has 0 frames: {video_path}")
                cap.release()
                # Return dummy frames
                dummy_frames = np.zeros((num_frames, 128, 128, 3), dtype=np.uint8)
                return dummy_frames
            
            # Calculate frame indices for uniform sampling
            if total_frames <= num_frames:
                # If video has fewer frames than needed, repeat frames
                frame_indices = list(range(total_frames)) * (num_frames // total_frames + 1)
                frame_indices = frame_indices[:num_frames]
            else:
                # Uniform sampling with stride
                step = max(1, total_frames // (num_frames * stride))
                frame_indices = list(range(0, total_frames, step))[:num_frames]
                
                # If we don't have enough frames, add more
                while len(frame_indices) < num_frames:
                    frame_indices.append(frame_indices[-1])
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Resize frame to target size for memory efficiency
                    frame = cv2.resize(frame, (128, 128))
                    frames.append(frame)
                else:
                    # If frame read fails, use previous frame or create dummy
                    if frames:
                        frames.append(frames[-1])
                    else:
                        dummy_frame = np.zeros((128, 128, 3), dtype=np.uint8)
                        frames.append(dummy_frame)
            
            cap.release()
            
            # Ensure we have exactly num_frames
            while len(frames) < num_frames:
                frames.append(frames[-1] if frames else np.zeros((128, 128, 3), dtype=np.uint8))
            
            frames = np.array(frames[:num_frames])
            
            # No disk saving - return frames directly
            return frames
            
        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")
            # Return dummy frames on error
            dummy_frames = np.zeros((num_frames, 128, 128, 3), dtype=np.uint8)
            return dummy_frames

    def _load_annotations_enhanced(self, video_id: str, frame_idx: int) -> Optional[np.ndarray]:
        """Load enhanced annotation for a specific frame"""
        annotation_path = os.path.join(self.annotations_dir, 'mat', video_id, f'{frame_idx:05d}.mat')
        if os.path.exists(annotation_path):
            try:
                mat_data = sio.loadmat(annotation_path)
                return mat_data
            except Exception as e:
                print(f"Warning: Could not load annotation {annotation_path}: {e}")
                return None
        return None
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get an enhanced sample with both action and actor labels"""
        row = self.data.iloc[idx]
        video_id = row['video_id']
        action = row['action']
        actor = row['actor']
        
        # Load video frames with enhanced extraction
        video_path = os.path.join(self.videos_dir, f"{video_id}.mp4")
        frames = self._extract_frames_enhanced(video_path, self.num_frames, self.temporal_stride,self.extract_frames_dir)
        
        # Apply enhanced temporal augmentation for training
        if self.mode == 'train' and config.temporal_augmentation:
            frames = self._apply_temporal_augmentation_enhanced(frames)
        
        # Apply spatial transformations
        if self.transform:
            transformed_frames = []
            for frame in frames:
                transformed = self.transform(image=frame)['image']
                transformed_frames.append(transformed)
            frames = torch.stack(transformed_frames)
        else:
            frames = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255.0
        
        # Convert labels to indices
        action_idx = self.action_to_idx[action]
        actor_idx = self.actor_to_idx[actor]
        
        return {
            'frames': frames,
            'action_label': torch.tensor(action_idx, dtype=torch.long),
            'actor_label': torch.tensor(actor_idx, dtype=torch.long),
            'action_name': action,
            'actor_name': actor,
            'video_id': video_id,
            'duration': row['duration'],
            'aspect_ratio': row['aspect_ratio']
        }
    
    def _apply_temporal_augmentation_enhanced(self, frames: np.ndarray) -> np.ndarray:
        """Apply enhanced temporal augmentation for training"""
        # Temporal reversal with probability
        if random.random() < 0.3:
            frames = frames[::-1]
        
        # Temporal jittering with adaptive shift
        if random.random() < 0.4:
            num_frames = len(frames)
            max_shift = min(3, num_frames // 4)  # Adaptive shift based on frame count
            shift = random.randint(-max_shift, max_shift)
            if shift > 0:
                frames = np.roll(frames, shift, axis=0)
            elif shift < 0:
                frames = np.roll(frames, shift, axis=0)
        
        # Frame dropping for robustness
        if random.random() < 0.2:
            num_frames = len(frames)
            drop_indices = random.sample(range(num_frames), num_frames // 8)
            for idx in drop_indices:
                if idx > 0 and idx < num_frames - 1:
                    # Interpolate dropped frame
                    frames[idx] = (frames[idx-1] + frames[idx+1]) // 2
        
        # Temporal stretching
        if random.random() < 0.1:
            # Slow motion effect
            frames = frames[::2]  # Take every other frame
            # Repeat frames to maintain length
            while len(frames) < self.num_frames:
                frames = np.concatenate([frames, frames[-1:]], axis=0)
            frames = frames[:self.num_frames]
        
        return frames


def get_enhanced_transforms(split: str = 'train') -> A.Compose:
    """Get optimized transforms for data augmentation"""
    if split == 'train':
        return A.Compose([
            # Basic spatial augmentations
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            
            # Color augmentations
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            
            # Geometric transformations
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
            
            # Resize and crop
            A.Resize(height=config.frame_size[0], width=config.frame_size[1]),
            A.RandomCrop(height=config.frame_size[0], width=config.frame_size[1], p=0.5),
            
            # Normalization
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        # Validation/Test transforms - only normalization
        return A.Compose([
            A.Resize(height=config.frame_size[0], width=config.frame_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def create_enhanced_data_loaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create enhanced data loaders for two-step classification"""
    # Create datasets
    train_dataset = EnhancedA2DDataset(
        csv_file=config.csv_file,
        videos_dir=config.videos_dir,
        annotations_dir=config.annotations_dir,
        transform=get_enhanced_transforms('train'),
        mode='train',
        num_frames=config.num_frames,
        temporal_stride=config.temporal_stride,
        frame_size=config.frame_size
    )
    
    val_dataset = EnhancedA2DDataset(
        csv_file=config.csv_file,
        videos_dir=config.videos_dir,
        annotations_dir=config.annotations_dir,
        transform=get_enhanced_transforms('val'),
        mode='val',
        num_frames=config.num_frames,
        temporal_stride=config.temporal_stride,
        frame_size=config.frame_size
    )
    
    test_dataset = EnhancedA2DDataset(
        csv_file=config.csv_file,
        videos_dir=config.videos_dir,
        annotations_dir=config.annotations_dir,
        transform=get_enhanced_transforms('val'),
        mode='test',
        num_frames=config.num_frames,
        temporal_stride=config.temporal_stride,
        frame_size=config.frame_size
    )
    
    # Create enhanced data loaders with better settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=True if config.num_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader


def _decode_label_static(label: int) -> Tuple[str, str]:
    """Static version of decode label for use outside the class"""
    if label == 0:
        return "background", "background"
    
    # Extract tens and ones digits
    tens = label // 10
    ones = label % 10
    
    # Map to action and actor
    action_map = {
        1: "climbing", 2: "crawling", 3: "eating", 4: "flying",
        5: "jumping", 6: "rolling", 7: "running", 8: "walking", 9: "none"
    }
    
    actor_map = {
        1: "adult", 2: "baby", 3: "ball", 4: "bird",
        5: "car", 6: "cat", 7: "dog"
    }
    
    action = action_map.get(ones, "background")
    actor = actor_map.get(tens, "background")
    
    return action, actor


def get_dataset_statistics():
    """Get simplified dataset statistics without loading samples"""
    try:
        # Load CSV data directly for statistics
        df = pd.read_csv(config.csv_file, header=None)
        df.columns = ['video_id', 'label', 'start_time', 'end_time', 
                     'height', 'width', 'num_frames', 'num_annotated_frames', 'usage']
        
        # Filter only valid labels
        df = df[df['label'].isin(config.valid_labels)]
        
        # Extract action and actor from label
        df['action'], df['actor'] = zip(*df['label'].apply(lambda x: _decode_label_static(x)))
        
        # Split data
        total_samples = len(df)
        val_split = config.val_split
        test_split = config.test_split
        
        train_size = int(total_samples * (1 - val_split - test_split))
        val_size = int(total_samples * val_split)
        test_size = total_samples - train_size - val_size
        
        # Count actions and actors from CSV
        action_counts = df['action'].value_counts().to_dict()
        actor_counts = df['actor'].value_counts().to_dict()
        
        print("Dataset Statistics:")
        print(f"Total samples: {total_samples}")
        print(f"Train samples: {train_size}")
        print(f"Validation samples: {val_size}")
        print(f"Test samples: {test_size}")
        print("\nAction distribution:")
        for action, count in sorted(action_counts.items()):
            print(f"  {action}: {count}")
        print("\nActor distribution:")
        for actor, count in sorted(actor_counts.items()):
            print(f"  {actor}: {count}")
        
        return {
            'train_samples': train_size,
            'val_samples': val_size,
            'test_samples': test_size,
            'action_counts': action_counts,
            'actor_counts': actor_counts
        }
    except Exception as e:
        print(f"Warning: Could not load dataset statistics: {e}")
        # Return default values
        return {
            'train_samples': 2469,
            'val_samples': 567,
            'test_samples': 746,
            'action_counts': {},
            'actor_counts': {}
        }
