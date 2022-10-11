import cv2
import time
import sys
import numpy as np
import shutil
import os
import re
import boto3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


def build_model(is_cuda):
    net = cv2.dnn.readNet("config_files/yolov5s.onnx")
    if is_cuda:
        print("Attempty to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4


def detect(image, net):
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False
    )
    net.setInput(blob)
    preds = net.forward()
    return preds


def load_classes():
    class_list = []
    with open("config_files/classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list


def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if classes_scores[class_id] > 0.25:

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes


def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


def upload(s3_bucket, file):
    if s3_bucket:
        s3 = boto3.resource("s3")
        bucket = s3.Bucket(s3_bucket)
        # upload
        bucket.upload_file(file, file)
        print(f"upload {file} done.")
    # remove
    os.remove(file)


def main(is_cuda, video, class_list, object_ids, s3_bucket):
    executor = ThreadPoolExecutor()

    colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

    net = build_model(is_cuda)
    capture = cv2.VideoCapture(video)

    start = time.time_ns()
    frame_count = 0
    total_frames = 0
    fps = -1

    is_save = False
    fmt = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None

    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    video_name = f"{timestamp}.mp4"

    while True:

        _, frame = capture.read()
        if frame is None:
            print("End of stream")
            break

        inputImage = format_yolov5(frame)
        outs = detect(inputImage, net)

        class_ids, confidences, boxes = wrap_detection(inputImage, outs[0])

        if len([c for c in class_ids if c in object_ids]) != 0:
            is_save = True

        frame_count += 1
        total_frames += 1

        for (classid, confidence, box) in zip(class_ids, confidences, boxes):
            color = colors[int(classid) % len(colors)]
            cv2.rectangle(frame, box, color, 2)
            cv2.rectangle(
                frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1
            )
            cv2.putText(
                frame,
                class_list[classid],
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
            )

        if frame_count >= 30:
            end = time.time_ns()
            fps = 1000000000 * frame_count / (end - start)
            frame_count = 0
            start = time.time_ns()

        if fps > 0:
            fps_label = "FPS: %.2f" % fps
            cv2.putText(
                frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

        cv2.imshow("output", frame)

        if writer is None and fps > 0:
            w, h = frame.shape[1], frame.shape[0]
            writer = cv2.VideoWriter(
                video_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
            )

        if writer:
            writer.write(frame)

            # update timestamp
            new_timestamp = datetime.now().strftime("%Y%m%d%H%M")
            if timestamp != new_timestamp:
                writer.release()
                writer = None
                if is_save:
                    print(f"save {video_name}")
                    executor.submit(upload, s3_bucket, video_name)
                else:
                    print(f"remove {video_name}")
                    os.remove(video_name)
                is_save = False
                timestamp = new_timestamp

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("finished by user")
            running = False
            if writer:
                writer.release()
            break

    print("Total frames: " + str(total_frames))


def check_objects(class_list, objects):
    ids = [i for i, v in enumerate(class_list) if v in objects]
    if len(ids) != len(objects):
        print(f"Error: unknown objects in {','.join(objects)}")
        return None
    return ids


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="video number", default=0)
    parser.add_argument("--cuda", help="Try to use CUDA", default=False)
    parser.add_argument(
        "--objects",
        help="When the objects are detected in the video, it is saved. The objects are specified with comma separated.",
        default="person",
    )
    parser.add_argument(
        "--s3-bucket",
        help="AWS s3 bucket name. If it is not specified, videos are not saved to AWS s3.",
        default=None,
    )
    args = parser.parse_args()

    class_list = load_classes()
    object_ids = check_objects(class_list, args.objects.split(","))
    if object_ids is None:
        parser.print_help()
        sys.exit(1)

    sys.exit(main(bool(args.cuda), args.video, class_list, object_ids, args.s3_bucket))
