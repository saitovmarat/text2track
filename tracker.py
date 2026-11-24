from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from utils import iou


@dataclass
class Track:
    bbox: tuple[float, float, float, float]
    missed: int = 0
    

class Tracker:
    def __init__(self, iou_thresh = 0.3):
        self.cur_tracks: dict[int, Track] = {} # {ID: bbox}
        self.next_id = 1
        self.iou_thresh = iou_thresh
    
    def update(self, bboxes) -> dict[int, Track]:
        if not self.cur_tracks:
            for bbox in bboxes:
                self.cur_tracks[self.next_id] = Track(bbox=bbox)
                self.next_id += 1
            return self.cur_tracks
        
        prev_ids = [id for id in self.cur_tracks.keys()]
        prev_bboxes = [track.bbox for track in self.cur_tracks.values()]
        iou_matrix = self.count_iou(prev_bboxes, bboxes)
        
        matched_cols = set()
        for row, col in linear_sum_assignment(iou_matrix, maximize=True):
            if iou_matrix[row, col] >= self.iou_thresh:
                cur_track = self.cur_tracks[prev_ids[row]]
                cur_track.bbox = bboxes[col]
                cur_track.missed = 0
                matched_cols.add(col)
                
        for col, bbox in enumerate(bboxes):
            if col not in matched_cols:
                self.cur_tracks[self.next_id] = Track(bbox=bbox)
                self.next_id += 1

        return self.cur_tracks
        
    @staticmethod
    def count_iou(bboxes_a: list[tuple], bboxes_b: list[tuple]):
        iou_matrix = np.empty((len(bboxes_a), len(bboxes_b)))
        for i, a in enumerate(bboxes_a):
            for j, b in enumerate(bboxes_b):
                iou_matrix[i, j] = iou(a, b)
        return iou_matrix
                    