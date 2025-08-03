import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import pathlib
import mimetypes
import imageio
import numpy as np
import argparse

from src.modules.pipeline import Pipeline
from src.utilities.vis_utils import render
from src.modules.results import EyeStateResultContainer
from src.commons.logger import Logger
from src.commons.get_weights import download_shape_predictor

logger = Logger()

def process_image(weights, predictor, path, **kwargs):
    detector_type = kwargs.get('detector', 'dlib')
    pipeline = Pipeline(
        weights=pathlib.Path(weights),
        shape_predictor=pathlib.Path(predictor),
        **kwargs
    )
    frame = cv2.imread(path)
    if frame is None:
        logger.info(f"Failed to read image: {path}")
        return
    results = pipeline.step(frame)
    output = render(frame, results, draw_landmarks=False, detector_type=detector_type)
    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", f"{os.path.basename(path)}")
    cv2.imwrite(out_path, output)
    logger.info(f"Saved: {out_path}")

def process_video(weights, predictor, path, **kwargs):
    detector_type = kwargs.get('detector', 'dlib')
    pipeline = Pipeline(
        weights=pathlib.Path(weights),
        shape_predictor=pathlib.Path(predictor),
        confidence_threshold=0.3,
        **kwargs
    )
    cap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", f"{os.path.basename(path)}")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width + 370, height))
    recent, window = [], 5
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        enhanced = enhance(frame)
        results = pipeline.step(enhanced)
        if results:
            recent.append(results)
            if len(recent) > window:
                recent.pop(0)
            smoothed = apply_smoothing(recent) if len(recent) >= 3 else results
        else:
            smoothed = results
        output = render(frame, smoothed, draw_landmarks=False, detector_type=detector_type)
        out.write(output)
        if frame_id % 30 == 0:
            logger.info(f"{frame_id}/{total} frames")

    cap.release()
    out.release()
    logger.info(f"Saved: {out_path}")

def process_gif(weights, predictor, path, **kwargs):
    detector_type = kwargs.get('detector', 'dlib')
    pipeline = Pipeline(
        weights=pathlib.Path(weights),
        shape_predictor=pathlib.Path(predictor),
        confidence_threshold=0.3,
        **kwargs
    )
    reader = imageio.get_reader(path)
    fps = reader.get_meta_data().get('duration', 0.1)
    total = len(reader)
    logger.info(f"Processing GIF: {total} frames")
    
    frame0 = reader.get_data(0)
    frame0 = (frame0 * 255).astype(np.uint8) if frame0.dtype != np.uint8 else frame0
    bgr = cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR)
    sample = render(bgr, EyeStateResultContainer([], [], [], [], np.empty((0,4)), np.empty((0,5,2)), np.empty((0,))), draw_landmarks=False, detector_type=detector_type)
    os.makedirs("output", exist_ok=True)
    writer = imageio.get_writer(os.path.join("output", f"{os.path.basename(path)}"), duration=fps, loop=0)
    recent, window = [], 3

    for i, frame in enumerate(reader):
        frame = (frame * 255).astype(np.uint8) if frame.dtype != np.uint8 else frame
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        enhanced = enhance(bgr)
        results = pipeline.step(enhanced)
        if results:
            recent.append(results)
            if len(recent) > window:
                recent.pop(0)
            smoothed = apply_smoothing(recent) if len(recent) >= 2 else results
        else:
            smoothed = results
        rendered = render(bgr, smoothed, draw_landmarks=False, detector_type=detector_type)
        rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb)
        if i % 10 == 0:
            logger.info(f"{i+1}/{total} frames")

    writer.close()
    logger.info(f"Saved: output/{os.path.basename(path)}")

def enhance(frame):
    blurred = cv2.GaussianBlur(frame.astype(np.float32), (3, 3), 0)
    lab = cv2.cvtColor(blurred.astype(np.uint8), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    lab = cv2.merge([l, a, b])
    contrast = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    sharpen = cv2.filter2D(contrast.astype(np.uint8), -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]))
    return np.clip(sharpen, 0, 255).astype(np.uint8)

def apply_smoothing(history):
    latest = history[-1]
    if not latest:
        return latest
    sm_left, sm_right, conf_l, conf_r = [], [], [], []
    for i in range(len(latest)):
        l_hist = [res.left_states[i] for res in history if i < len(res)]
        r_hist = [res.right_states[i] for res in history if i < len(res)]
        l_conf = [res.left_confidences[i] for res in history if i < len(res)]
        r_conf = [res.right_confidences[i] for res in history if i < len(res)]
        sm_left.append(weighted_vote(l_hist, l_conf))
        sm_right.append(weighted_vote(r_hist, r_conf))
        conf_l.append(np.mean(l_conf))
        conf_r.append(np.mean(r_conf))
        
    return EyeStateResultContainer(
        left_states=sm_left,
        right_states=sm_right,
        left_confidences=conf_l,
        right_confidences=conf_r,
        bboxes=latest.bboxes,
        landmarks=latest.landmarks,
        scores=latest.scores
    )

def weighted_vote(states, confs):
    if not states:
        return "open"
    scores = {}
    for s, c in zip(states, confs):
        scores[s] = scores.get(s, 0) + c
    return max(scores, key=scores.get)

def run_webcam(weights, predictor, cam=0, **kwargs):
    detector_type = kwargs.get('detector', 'dlib')
    pipeline = Pipeline(
        weights=pathlib.Path(weights),
        shape_predictor=pathlib.Path(predictor),
        **kwargs
    )
    cap = cv2.VideoCapture(cam)
    logger.info("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = pipeline.step(frame)
        output = render(frame, results, draw_landmarks=False, detector_type=detector_type)
        cv2.imshow("Eye State", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Eye State Detection System")
    parser.add_argument("input", help="Input file path or 'webcam'")
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--weights", default="./weights/eye_state_classifier.h5")
    parser.add_argument("--detector", choices=["dlib", "retinaface"], default="dlib", help="Face detector to use")
    parser.add_argument("--confidence_threshold", type=float, help="Confidence threshold for face detection")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu/gpu)")
    args = parser.parse_args()

    if args.device.lower() == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    predictor_path = download_shape_predictor()
    
    kwargs = {}
    kwargs['detector'] = args.detector
    kwargs['device'] = args.device
    if args.confidence_threshold is not None:
        kwargs['confidence_threshold'] = args.confidence_threshold

    if args.input.lower() == "webcam":
        run_webcam(args.weights, predictor_path, args.camera_id, **kwargs)
    else:
        if not os.path.exists(args.input):
            logger.info(f"File not found: {args.input}")
            return

        mime, _ = mimetypes.guess_type(args.input)
        if mime and "image" in mime and args.input.lower().endswith(".gif"):
            process_gif(args.weights, predictor_path, args.input, **kwargs)
        elif mime and "image" in mime:
            process_image(args.weights, predictor_path, args.input, **kwargs)
        elif mime and "video" in mime:
            process_video(args.weights, predictor_path, args.input, **kwargs)
        else:
            logger.info("Unsupported file type")

if __name__ == "__main__":
    main()
