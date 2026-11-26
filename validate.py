from pathlib import Path
from collections import defaultdict
import csv
from configparser import ConfigParser
from contextlib import redirect_stdout
from io import StringIO
from models.tracker import Track, Tracker
import sys
from unittest.mock import patch
import numpy as np

trackeval_path = Path(__file__).resolve().parent.parent / "TrackEval"
sys.path.insert(0, str(trackeval_path))
import trackeval


VIDEO_NAME = "MOT17-13-FRCNN"
DATA_PATH = Path(f"./data/MOT17/train/{VIDEO_NAME}")

OUTPUT_PATH = Path("results/iou_tracker/data") 
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
preds_path = OUTPUT_PATH / f"{VIDEO_NAME}.txt"
        
def get_preds() -> dict:
    labels = defaultdict(list)
    with open(DATA_PATH / "det" / "det.txt", encoding="utf-8") as f:
        for row in csv.reader(f):
            frame_number = int(row[0])
            box = tuple(map(float, row[2:6]))
            labels[frame_number].append(box)
    return labels

config = ConfigParser()
with open(DATA_PATH / "seqinfo.ini", encoding="utf-8") as f:
    config.read_file(f)
sequence = config["Sequence"]

tracker = Tracker(iou_thresh=0.3, max_missed=5)
labels = get_preds()

with preds_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for frame_idx in range(1, sequence.getint("seqLength") + 1):
        tracks = tracker.update(labels[frame_idx])
        for track_id, track in tracks.items():
            if track.missed > 0:
                continue
            
            writer.writerow([frame_idx, track_id, *track.cur_bbox, 1, -1, -1, -1])

dataset = trackeval.datasets.MotChallenge2DBox({
    "GT_FOLDER": str(DATA_PATH.parent),
    "TRACKERS_FOLDER": "results",
    "TRACKERS_TO_EVAL": ["iou_tracker"],
    "SEQ_INFO": {VIDEO_NAME: None},
    "SKIP_SPLIT_FOL": True,
    "PRINT_CONFIG": False,
})

evaluator = trackeval.Evaluator({
    "OUTPUT_SUMMARY": False,
    "OUTPUT_DETAILED": False,
    "PRINT_RESULTS": False,   # скрыть таблицы метрик в консоли
    "PRINT_CONFIG": False,
    "TIME_PROGRESS": False,
    "PLOT_CURVES": True,
})

metrics = [
    trackeval.metrics.HOTA(),
    trackeval.metrics.CLEAR({"PRINT_CONFIG": False}),
    trackeval.metrics.Identity({"PRINT_CONFIG": False}),
]

with (
    patch.object(np, "int", int, create=True),
    patch.object(np, "float", float, create=True),
    redirect_stdout(StringIO()),
):
    results, _ = evaluator.evaluate([dataset], metrics)
    
tracker_results = results[dataset.get_name()]["iou_tracker"]
scores = tracker_results[VIDEO_NAME]["pedestrian"]

print(f"IDF1: {scores['Identity']['IDF1']:.2%}")
print(f"AssA: {scores['HOTA']['AssA'].mean():.2%}")
print(f"IDSW: {scores['CLEAR']['IDSW']}")
print(f"HOTA: {scores['HOTA']['HOTA'].mean():.2%}")
