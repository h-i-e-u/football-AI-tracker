class FallDetector:

    def __init__(self, ratio_threshold=1.2, min_frames=5):

        self.ratio_threshold = ratio_threshold
        self.min_frames = min_frames

        self.states = {}


    def update(self, tracker_id, box):

        x1, y1, x2, y2 = box

        w = x2 - x1
        h = y2 - y1

        ratio = w / (h + 1e-6)


        # bbox is considered as fallen if the width is greater than the height by a certain ratio
        if ratio > self.ratio_threshold:

            self.states[tracker_id] = (
                self.states.get(tracker_id, 0) + 1
            )

        else:
            self.states[tracker_id] = 0


        if self.states[tracker_id] >= self.min_frames:
            return True

        return False