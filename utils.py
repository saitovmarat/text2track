def iou(bbox_a, bbox_b):
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b
    
    left = max(ax, bx)
    right = min(ax + aw, bx + bw)
    top = max(ay, by)
    bottom = min(ay + ah, by + bh)
    
    intersection_area = max(0, right - left) * max(0, bottom - top)
    total_area = aw * ah + bw * bh - intersection_area
    return intersection_area / total_area