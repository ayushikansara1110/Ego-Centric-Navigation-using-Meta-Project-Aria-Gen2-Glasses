from dataclasses import dataclass
from ultralytics import YOLO

PERSON_CLASS_ID = 0          # COCO 'person'
DOOR_CLASS_ID = None         # set this once you've fine-tuned / loaded a door-capable model

COLLISION_DISTANCE_M = 1.5   # warn if a person is estimated closer than this
COLLISION_CONE_FRAC = 0.35   # fraction of frame width counted as "directly ahead"
ASSUMED_PERSON_HEIGHT_M = 1.7


@dataclass
class DoorEvent:
    side: str          # "left" or "right"
    confidence: float

@dataclass
class PersonEvent:
    side: str          # "left", "right", "ahead"
    distance_m: float
    is_collision_risk: bool


class Perception:
    def __init__(self, model_path="yolov8n.pt", focal_length_px=None):
        self.model = YOLO(model_path)
        self.focal_length_px = focal_length_px  # from Aria RGB camera calibration

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        mid_x = w / 2
        results = self.model(frame, verbose=False)[0]

        doors, people = [], []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center_x = (x1 + x2) / 2
            side = "right" if center_x > mid_x else "left"

            if DOOR_CLASS_ID is not None and cls_id == DOOR_CLASS_ID:
                doors.append(DoorEvent(side=side, confidence=conf))

            elif cls_id == PERSON_CLASS_ID:
                bbox_height_px = y2 - y1
                distance_m = self._estimate_distance(bbox_height_px)

                # "ahead" if inside the central collision cone, else left/right
                if abs(center_x - mid_x) < (w * COLLISION_CONE_FRAC / 2):
                    person_side = "ahead"
                else:
                    person_side = side

                is_risk = (
                    distance_m is not None
                    and distance_m < COLLISION_DISTANCE_M
                    and person_side == "ahead"
                )
                people.append(PersonEvent(side=person_side, distance_m=distance_m, is_collision_risk=is_risk))

        return doors, people

    def _estimate_distance(self, bbox_height_px):
        """Rough monocular distance from apparent height. Good enough for a
        near/far warning, not precision ranging. Swap in Aria depth if/when
        you have it."""
        if self.focal_length_px is None or bbox_height_px <= 0:
            return None
        return (ASSUMED_PERSON_HEIGHT_M * self.focal_length_px) / bbox_height_px
