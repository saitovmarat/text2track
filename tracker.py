from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from utils import iou


BBox = tuple[float, float, float, float]

@dataclass
class Track:
    cur_frame_id: int
    cur_bbox: BBox
    prev_frame_id: int = None
    prev_bbox: BBox = None
    missed: int = 0
    
    def update(self, bbox: BBox, frame_id: int = None):        
        self.prev_bbox = self.cur_bbox
        self.prev_frame_id = self.cur_frame_id
        self.cur_bbox = bbox
        self.cur_frame_id = frame_id
        self.missed = 0
        
    def predict_bbox(self, frame_id: int) -> BBox:
        if self.prev_bbox is None or self.prev_frame_id is None:
            return self.cur_bbox
        
        px, py, _, _ = self.prev_bbox
        x, y, w, h = self.cur_bbox
        
        frames_between = self.cur_frame_id - self.prev_frame_id
        vx = (x - px) / frames_between
        vy = (y - py) / frames_between
        
        frames_ahead = frame_id - self.cur_frame_id
        return x + vx * frames_ahead, y + vy * frames_ahead, w, h
        
        
class Tracker:
    def __init__(self, iou_thresh = 0.3, max_missed = 2):
        self.cur_tracks: dict[int, Track] = {} # {ID: bbox}
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self.next_id = 1
        self.frame_idx = 0
    
    def update(self, bboxes) -> dict[int, Track]:
        self.frame_idx += 1
        if not self.cur_tracks:
            for bbox in bboxes:
                self._add_new_track(bbox)
            return self.cur_tracks
        
        for track in self.cur_tracks.values():
            track.missed += 1            

        prev_ids = [id for id in self.cur_tracks.keys()]
        prev_bboxes = [
            track.predict_bbox(self.frame_idx) 
            for track in self.cur_tracks.values()
        ]
        iou_matrix = self.count_iou(prev_bboxes, bboxes)
        
        matched_cols = set()
        rows, cols = linear_sum_assignment(iou_matrix, maximize=True)
        for row, col in zip(rows, cols):
            if iou_matrix[row, col] >= self.iou_thresh:
                self.cur_tracks[prev_ids[row]].update(bboxes[col], self.frame_idx)
                matched_cols.add(col)
                
        for col, bbox in enumerate(bboxes):
            if col not in matched_cols:
                self._add_new_track(bbox)

        for track_id in prev_ids:
            if self.cur_tracks[track_id].missed > self.max_missed:
                del self.cur_tracks[track_id]

        return self.cur_tracks
    
    def _add_new_track(self, bbox):
        self.cur_tracks[self.next_id] = Track(
            cur_bbox=bbox, cur_frame_id=self.frame_idx
        )
        self.next_id += 1


    @staticmethod
    def count_iou(bboxes_a: list[tuple], bboxes_b: list[tuple]):
        iou_matrix = np.empty((len(bboxes_a), len(bboxes_b)))
        for i, a in enumerate(bboxes_a):
            for j, b in enumerate(bboxes_b):
                iou_matrix[i, j] = iou(a, b)
        return iou_matrix
                    