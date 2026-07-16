import numpy as np
import cv2

class TeamTracker:
    def __init__(self, ema_alpha=0.08):
        self.cents, self.alpha = None, ema_alpha

    def assign(self, colors):
        v_idx = [i for i, c in enumerate(colors) if c]
        res, fall = [-1] * len(colors), [(220, 80, 60), (60, 80, 220)]
        if len(v_idx) < 2: return res, fall

        pts = np.array([colors[i] for i in v_idx], dtype=np.float32)
        if self.cents is None:
            _, _, self.cents = cv2.kmeans(pts, 2, None, (3, 20, 0.5), 5, cv2.KMEANS_PP_CENTERS)

        lbls = np.linalg.norm(pts[:, None] - self.cents, axis=2).argmin(axis=1)
        for t in range(2):
            if (m := (lbls == t)).any():
                self.cents[t] = (1 - self.alpha) * self.cents[t] + self.alpha * pts[m].mean(0)
                
        for k, i in enumerate(v_idx): res[i] = int(lbls[k])
        return res, [tuple(map(int, c)) for c in self.cents]

def jersey_color(img, x1, y1, x2, y2):
    if not (crop := img[y1:(y1+y2)//2, x1:x2]).size: return None
    mask = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), (35, 40, 40), (85, 255, 255)).ravel() == 0
    ng = crop.reshape(-1, 3)[mask]
    return tuple(map(int, ng.mean(0))) if len(ng) >= 20 else None