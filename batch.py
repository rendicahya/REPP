import json
import warnings
from pathlib import Path

from assertpy.assertpy import assert_that
from python_config import Config
from python_file import count_files
from REPP import REPP
from repp_utils import get_video_frame_iterator
from tqdm import tqdm

conf = Config("../config.json")
input_dir = Path.cwd().parent / conf.repp.input.path
output_dir = Path.cwd().parent / conf.repp.output.path
repp_conf = conf.repp.configuration
n_files = count_files(input_dir, ext=".pckl")
repp_params = json.load(open(repp_conf, "r"))
repp = REPP(**repp_params, store_coco=True)

assert_that(input_dir).is_directory().is_readable()
assert_that(repp_conf).is_file().is_readable()
warnings.filterwarnings("ignore")

bar = tqdm(total=n_files)

for file in input_dir.glob("**/*.pckl"):
    bar.set_description(file.name)

    action = file.parent.name
    total_preds = []

    for vid, video_preds in get_video_frame_iterator(file):
        preds_coco, _ = repp(video_preds)
        total_preds += preds_coco

    preds_out_path = output_dir / action / file.with_suffix(".json").name

    preds_out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(total_preds, open(preds_out_path, "w"))
    bar.update(1)

bar.close()
