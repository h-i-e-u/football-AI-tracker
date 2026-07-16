import math

class PossessionTracker:
    def __init__(self, proximity_px=80):
        self.prox = proximity_px
        self.counts = [0, 0]

    def update(self, b_box, p_boxes, t_ids):
        if b_box is None:
            return -1

        bx = (b_box[0] + b_box[2]) / 2
        by = (b_box[1] + b_box[3]) / 2

        dist, tid = min(
            (
                (
                    math.hypot(
                        bx - max(x1, min(x2, bx)),
                        by - max(y1, min(y2, by))
                    ),
                    team
                )
                for (x1, y1, x2, y2), team in zip(p_boxes, t_ids)
                if team >= 0
            ),
            default=(float("inf"), -1)
        )

        if dist <= self.prox:
            self.counts[tid] += 1
            return tid

        return -1

    def percentages(self):
        total = sum(self.counts)
        if total == 0:
            return [50, 50]
        return [100 * c / total for c in self.counts]