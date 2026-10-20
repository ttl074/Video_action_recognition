import os
import shutil
import video_datasets as vd
import numpy as np
from PIL import Image
from utils import transform_stats, compose_data_transforms
import torch

def test_train_transform_frame_consistency(tmp_path,n_frames):
    """
    Test that train transforms apply consistently to all frames
    """
    rng = np.random.default_rng(2026)
    torch.manual_seed(2026)
    arr = rng.integers(0,256,size=(60,80,3),dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(tmp_path / "frame.jpg")
    frames_dir = os.path.join(tmp_path,"frames")
    os.makedirs(frames_dir)
    for i in range(n_frames):
        shutil.copy(os.path.join(tmp_path,"frame.jpg"),os.path.join(frames_dir,f"{i:05d}.jpg"))
    h, w, mean, std = transform_stats("lrcn")
    tr_transforms,_ = compose_data_transforms(h, w, mean, std)
    dataset = vd.VideoDataset([(frames_dir,0)],n_frames,tr_transforms)
    clips = []
    failures = []
    for i in range(10):
        clip,label = dataset[0]
        if clip.shape != (n_frames,3,h,w):
            failures.append(f"iter {i}: shape {tuple(clip.shape)}, expected {(n_frames, 3, h, w)}")
        if clip.dtype != torch.float32:
            failures.append(f"iter {i}: dtype {clip.dtype}, expected torch.float32")
        if label != 0:
            failures.append(f"iter {i}: label {label}, expected 0")
        for j in range(1,n_frames):
            if not torch.equal(clip[0],clip[j]):
                failures.append(f"iter {i}: frame {j} differs from frame 0")
        clips.append(clip)
    assert not failures, f"{len(failures)} check(s) failed, e.g. {failures[:5]}"
    assert any(not torch.equal(clips[0],c) for c in clips[1:]),"all 10 clips were identical; random augmentation may not be running"


def test_data_leaks(ucf50_frames_dir):
    """
    Test to make sure that splits do not have overlapping groups and that none are empty
    """
    vid_dataset,_= vd.load_dataset(ucf50_frames_dir)
    splits = vd.dataset_split(vid_dataset,0.75,0.15)
    sets = [set(),set(),set()]
    for i in range(len(splits)):
        for path,_ in splits[i]:
            g = vd.get_group(path) 
            sets[i].add(g)
    num_groups = len(sets[0])+len(sets[1])+len(sets[2])
    assert sets[0], "Training set is empty!"
    assert sets[1], "Validation set is empty!"
    assert sets[2], "Test set is empty!"
    assert not sets[0] & sets[1], f"Training and validation share groups {sorted(sets[0] & sets[1])}"
    assert not sets[1] & sets[2], f"Validation and test share groups {sorted(sets[1] & sets[2])}"
    assert not sets[0] & sets[2], f"Training and Test share groups {sorted(sets[0] & sets[2])}"
    assert num_groups == 25, f"UCF50 has 25 groups found {num_groups}"

