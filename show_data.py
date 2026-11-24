import cv2
from pathlib import Path
from collections import defaultdict
import csv
from configparser import ConfigParser
import random
from time import perf_counter


video_name = "MOT17-02-FRCNN"
DATA_PATH = Path(f"./data/MOT17/train/{video_name}")
frames_path = DATA_PATH / "img1"


def color_by_id(track_id=1):
    rng = random.Random(track_id)
    return tuple(rng.randint(60, 225) for _ in range(3))
    
def draw_bboxes(image, labels):
    for track_id, bbox in labels:
        x, y, w, h = bbox
        color = color_by_id(track_id)
        left, top = round(x) - 1, round(y) - 1
        right, bottom = left + round(w) - 1, top + round(h) - 1 
        
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        cv2.putText(
            image, str(track_id), (max(0, left), max(18, top - 6)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
        )              
        
def get_preds() -> dict:
    labels = defaultdict(list)
    with open(DATA_PATH / "det" / "det.txt", encoding="utf-8") as f:
        for row in csv.reader(f):
            frame_number = int(row[0])
            track_id = int(row[1])
            box = tuple(map(float, row[2:6]))
            labels[frame_number].append((track_id, box))
    return labels

config = ConfigParser()
with open(DATA_PATH / "seqinfo.ini", encoding="utf-8") as f:
    config.read_file(f)
sequence = config["Sequence"]
fps = sequence.getint("frameRate")


labels = get_preds()
start_time = perf_counter()
for frame_idx in range(1, sequence.getint("seqLength") + 1):
    image = cv2.imread(frames_path / f"{frame_idx:06d}.jpg")
    draw_bboxes(image, labels[frame_idx])
    cv2.imshow(video_name, image)
    
    next_frame_time = start_time + frame_idx / fps
    delay_ms = max(1, round((next_frame_time - perf_counter()) * 1000))
    if cv2.waitKey(delay_ms) in (27, ord("q")):
        break

cv2.destroyAllWindows()