import random
import cv2
from tracker import Track
from collections import defaultdict


def color_by_id(track_id=1):
    rng = random.Random(track_id)
    return tuple(rng.randint(60, 225) for _ in range(3))
    

def draw_bboxes(image, labels: dict):
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
        
        
def draw_tracks(image, tracks: dict[int, Track]):
    for track_id, track in tracks.items():
        x, y, w, h = track.cur_bbox
        color = color_by_id(track_id)
        left, top = round(x) - 1, round(y) - 1
        right, bottom = left + round(w) - 1, top + round(h) - 1 
        
        cv2.rectangle(image, (left, top), (right, bottom), color, 2)
        cv2.putText(
            image, str(track_id), (max(0, left), max(18, top - 6)), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
        )  