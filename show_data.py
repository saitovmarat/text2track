import cv2
from pathlib import Path
from collections import defaultdict
import csv
from configparser import ConfigParser
from time import perf_counter
from tracker import Track, Tracker
from visualisation import draw_tracks


video_name = "MOT17-02-FRCNN"
DATA_PATH = Path(f"./data/MOT17/train/{video_name}")
frames_path = DATA_PATH / "img1"
        
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
fps = sequence.getint("frameRate")

tracker = Tracker(iou_thresh=0.3, max_missed=5)
labels = get_preds()
start_time = perf_counter()
for frame_idx in range(1, sequence.getint("seqLength") + 1):
    image = cv2.imread(frames_path / f"{frame_idx:06d}.jpg")
    tracks = tracker.update(labels[frame_idx])
    draw_tracks(image, tracks)
    cv2.imshow(video_name, image)
    
    next_frame_time = start_time + frame_idx / fps
    delay_ms = max(1, round((next_frame_time - perf_counter()) * 1000))
    if cv2.waitKey(delay_ms) in (27, ord("q")):
        break

cv2.destroyAllWindows()