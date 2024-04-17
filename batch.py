import json
import warnings
from pathlib import Path

import cv2
import numpy as np
from assertpy.assertpy import assert_that
from config import settings as conf
from python_file import count_files
from python_video import frames_to_video, video_frames, video_info
from REPP import REPP
from repp_utils import get_video_frame_iterator
from tqdm import tqdm

root = Path.cwd().parent
dataset = conf.active.dataset
detector = conf.active.detector
mode = conf.active.mode
relevancy_model = conf.relevancy.active.method
relevancy_threshold = conf.relevancy.active.threshold
smoothing = conf.active.smooth_mask.enabled

object_selection = conf.active.object_selection
method = "select" if object_selection else "detect"
method_dir = root / "data" / dataset / detector / method

if method == "detect":
    pckl_in_dir = method_dir / "dump"
    mask_out_dir = method_dir / "REPP/mask"
elif method == "select":
    pckl_in_dir = method_dir / mode / "dump"
    mask_out_dir = method_dir / mode / "REPP/mask"

    if mode == "intercutmix":
        pckl_in_dir = pckl_in_dir / relevancy_model / relevancy_threshold
        mask_out_dir = mask_out_dir / relevancy_model / relevancy_threshold

repp_conf = conf.repp.configuration
repp_params = json.load(open(repp_conf, "r"))
repp = REPP(**repp_params, store_coco=True)
generate_videos = conf.repp.output.video.generate
video_in_ext = conf[dataset].ext
video_in_dir = root / conf[dataset].path
gaussian_size = conf.active.smooth_mask.gaussian_size

print("Dataset:", dataset)
print("Mode:", mode)
print("Object selection:", object_selection)
print("Generate videos:", generate_videos)
print("Relevancy model:", relevancy_model)
print("Relevancy thresh.:", relevancy_threshold)
print("Input:", pckl_in_dir)
print("Output:", mask_out_dir)

assert_that(pckl_in_dir).is_directory().is_readable()
assert_that(repp_conf).is_file().is_readable()

if generate_videos:
    video_out_dir = root / "data" / dataset / "REPP" / mode / "videos"
    video_out_ext = conf.repp.output.video.ext

    assert_that(video_in_dir).is_directory().is_readable()
    assert_that(video_out_ext).is_type_of(str).matches(r"^\.[a-zA-Z0-9]{3}$")

warnings.filterwarnings("ignore")

n_files = count_files(pckl_in_dir, ext=".pckl")
bar = tqdm(total=n_files)

for pckl in pckl_in_dir.glob("**/*.pckl"):
    action = pckl.parent.name
    total_preds = []

    video_name = pckl.with_suffix(video_in_ext).name
    video_path = video_in_dir / action / video_name

    if not video_path.exists():
        print("Video not found:", video_path)
        continue

    vid_info = video_info(video_path)
    vid_height, vid_width = vid_info["height"], vid_info["width"]
    n_frames = vid_info["n_frames"]
    mask_cube = np.zeros((n_frames, vid_height, vid_width), np.uint8)
    mask_out_path = mask_out_dir / action / pckl.stem

    if generate_videos:
        frames = video_frames(video_path, reader=conf.active.video.reader)
        out_frames = []

    for vid, video_preds in get_video_frame_iterator(pckl):
        preds_coco, _ = repp(video_preds)
        total_preds += preds_coco

    for f in range(n_frames):
        boxes = [item["bbox"] for item in total_preds if int(item["image_id"]) == f]

        if generate_videos:
            frame = next(frames)

        for box in boxes:
            x1, y1, w, h = [round(v) for v in box]
            x2, y2 = x1 + w, y1 + h
            mask_cube[f, y1:y2, x1:x2] = 255

            if smoothing:
                mask_cube[f] = cv2.GaussianBlur(
                    mask_cube[f], (gaussian_size, gaussian_size), 0
                )

            if generate_videos:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if generate_videos:
            out_frames.append(frame)

    mask_out_path.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(mask_out_path, mask_cube)

    if generate_videos:
        video_out_path = video_out_dir / action / pckl.with_suffix(video_out_ext).name

        video_out_path.parent.mkdir(parents=True, exist_ok=True)
        frames_to_video(
            frames=out_frames,
            target=video_out_path,
            writer=conf.active.video.writer,
            fps=vid_info["fps"],
        )

    bar.update(1)

bar.close()
