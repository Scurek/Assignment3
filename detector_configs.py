import detectors.cascade_detector.detector as cascade_detector
import detectors.yolo3.detector as yolo3_detector

detectors_yolo = {
        "Cascade": {
            "detector": cascade_detector.Detector(1.10, 1),
            "color": (0, 255, 0),
            "predictions": [],
            "iou_arr": [],
            "types": [0, 0, 0]
        },
        "Tiny300": {
            "detector": yolo3_detector.Detector("Tiny300"),
            "color": (0, 0, 255),
            "predictions": [],
            "iou_arr": [],
            "types": [0, 0, 0]
        },
        "Tiny300_ConfTresh-0.5": {
            "detector": yolo3_detector.Detector("Tiny300", 0.5),
            "color": (0, 0, 255),
            "predictions": [],
            "iou_arr": [],
            "types": [0, 0, 0]
        },
        "Tiny50": {
            "detector": yolo3_detector.Detector("Tiny50"),
            "color": (0, 0, 255),
            "predictions": [],
            "iou_arr": [],
            "types": [0, 0, 0]
        },
        "Tiny50_ConfTresh-0.5": {
            "detector": yolo3_detector.Detector("Tiny50", 0.5),
            "color": (0, 0, 255),
            "predictions": [],
            "iou_arr": [],
            "types": [0, 0, 0]
        },
        "Norm50": {
            "detector": yolo3_detector.Detector("Norm50"),
            "color": (0, 255, 255),
            "predictions": [],
            "iou_arr": [],
            "types": [0, 0, 0]
        },
        "Norm50_ConfTresh-0.5": {
            "detector": yolo3_detector.Detector("Norm50", 0.5),
            "color": (0, 255, 255),
            "predictions": [],
            "iou_arr": [],
            "types": [0, 0, 0]
        },
        "spp50": {
            "detector": yolo3_detector.Detector("Spp50"),
            "color": (255, 255, 0),
            "predictions": [],
            "iou_arr": [],
            "types": [0, 0, 0]
        },
        "Spp50_ConfTresh-0.5": {
            "detector": yolo3_detector.Detector("Spp50", 0.5),
            "color": (255, 255, 0),
            "predictions": [],
            "iou_arr": [],
            "types": [0, 0, 0]
        }
}