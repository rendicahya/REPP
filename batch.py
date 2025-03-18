import sys

sys.path.append(".")

import json
import warnings
from pathlib import Path

import click
import cv2
import mmcv
import numpy as np
from repp_utils import get_video_frame_iterator
from tqdm import tqdm

from assertpy.assertpy import assert_that
from config import settings as conf
from python_file import count_files
from python_video import frames_to_video
from REPP import REPP

ROOT = Path.cwd()
DATASET = conf.active.dataset
DETECTOR = conf.active.detector
DET_CONF = str(conf.unidet.detect.confidence)
METHOD = conf.active.mode
USE_REPP = conf.active.USE_REPP
RELEV_MODEL = conf.active.relevancy.method
RELEV_THRESH = str(conf.active.relevancy.threshold)
SMOOTHING = conf.active.smooth_mask.enabled

MID_DIR = ROOT / "data" / DATASET / DETECTOR / DET_CONF / METHOD
video_out_ext = conf.repp.output.video.ext

if METHOD in ("allcutmix", "actorcutmix"):
    PCKL_IN_DIR = MID_DIR / "dump"
    MASK_OUT_DIR = MID_DIR / "REPP/mask"
    VIDEO_OUT_DIR = MID_DIR / "REPP/videos"
else:
    PCKL_IN_DIR = MID_DIR / "dump" / RELEV_MODEL / RELEV_THRESH
    MASK_OUT_DIR = (
        MID_DIR / ("REPP/mask" if USE_REPP else "mask") / RELEV_MODEL / RELEV_THRESH
    )
    VIDEO_OUT_DIR = (
        MID_DIR / ("REPP/videos" if USE_REPP else "videos") / RELEV_MODEL / RELEV_THRESH
    )

REPP_CONF = conf.repp.configuration
repp_params = json.load(open(REPP_CONF, "r"))
repp = REPP(**repp_params, store_coco=True)
GENERATE_VIDEOS = conf.repp.output.video.generate
VIDEO_EXT = conf[DATASET].ext
VIDEO_IN_DIR = ROOT / conf[DATASET].path
GAUSSIAN_SIZE = conf.active.smooth_mask.GAUSSIAN_SIZE

assert_that(PCKL_IN_DIR).is_directory().is_readable()
assert_that(REPP_CONF).is_file().is_readable()

print("Input:", PCKL_IN_DIR.relative_to(ROOT))
print(
    "Output:",
    MASK_OUT_DIR.relative_to(ROOT),
    "(exists)" if MASK_OUT_DIR.exists() else "(not exists)",
)

if GENERATE_VIDEOS:
    assert_that(VIDEO_IN_DIR).is_directory().is_readable()
    assert_that(video_out_ext).is_type_of(str).matches(r"^\.[a-zA-Z0-9]{3}$")

    print(f"Video output: {VIDEO_OUT_DIR.relative_to(ROOT)}")

if not click.confirm("\nDo you want to continue?", show_default=True):
    exit("Aborted.")

warnings.filterwarnings("ignore")

n_files = count_files(PCKL_IN_DIR, ext=".pckl")
bar = tqdm(total=n_files, dynamic_ncols=True)

for pckl in PCKL_IN_DIR.glob("**/*.pckl"):
    action = pckl.parent.name
    total_preds = []

    video_name = pckl.with_suffix(VIDEO_EXT).name
    video_path = VIDEO_IN_DIR / action / video_name

    if not video_path.exists():
        print("Video not found:", video_path)
        continue

    video_reader = mmcv.VideoReader(str(video_path))
    vid_width, vid_height = video_reader.resolution
    n_frames = video_reader.frame_cnt
    mask_cube = np.zeros((n_frames, vid_height, vid_width), np.uint8)
    mask_out_path = MASK_OUT_DIR / action / pckl.stem
    out_frames = []

    for vid, video_preds in get_video_frame_iterator(pckl):
        preds_coco, _ = repp(video_preds)
        total_preds += preds_coco

    for f in range(n_frames):
        boxes = [item["bbox"] for item in total_preds if int(item["image_id"]) == f]

        if GENERATE_VIDEOS:
            frame = video_reader.read()

        for box in boxes:
            x1, y1, w, h = [round(v) for v in box]
            x2, y2 = x1 + w, y1 + h
            mask_cube[f, y1:y2, x1:x2] = 255

            if SMOOTHING:
                mask_cube[f] = cv2.GaussianBlur(
                    mask_cube[f], (GAUSSIAN_SIZE, GAUSSIAN_SIZE), 0
                )

            if GENERATE_VIDEOS:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        if GENERATE_VIDEOS:
            out_frames.append(frame)

    mask_out_path.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(mask_out_path, mask_cube)

    if GENERATE_VIDEOS:
        video_out_path = VIDEO_OUT_DIR / action / pckl.with_suffix(video_out_ext).name

        video_out_path.parent.mkdir(parents=True, exist_ok=True)
        frames_to_video(
            frames=out_frames,
            target=video_out_path,
            writer=conf.active.video.writer,
            fps=video_reader.fps,
        )

    bar.update(1)

bar.close()
