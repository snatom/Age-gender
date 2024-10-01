# Import required modules
import cv2 as cv
import math
import time
import argparse
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def getFaceBox(net, frame, conf_threshold=0.7):
    """
    Detect faces in the frame and return bounding boxes
    """
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    
    return frameOpencvDnn, bboxes

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
    parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
    parser.add_argument('--output', help='Path to save the output video.')
    parser.add_argument('--conf_threshold', type=float, default=0.7, help='Confidence threshold for face detection.')
    
    args = parser.parse_args()

    # File paths for the model files
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    # Constants
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    # Load networks
    ageNet = cv.dnn.readNet(ageModel, ageProto)
    genderNet = cv.dnn.readNet(genderModel, genderProto)
    faceNet = cv.dnn.readNet(faceModel, faceProto)

    # Open video file, image file, or capture from camera
    cap = cv.VideoCapture(args.input if args.input else 0)
    if not cap.isOpened():
        logging.error("Error: Unable to open video capture.")
        return

    # Prepare output video file if provided
    if args.output:
        output_writer = None
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        output_writer = cv.VideoWriter(args.output, cv.VideoWriter_fourcc(*'XVID'), 20, (frame_width, frame_height))
        logging.info(f"Output will be saved to {args.output}")

    padding = 20

    while cv.waitKey(1) < 0:
        # Read a frame from the video/camera
        hasFrame, frame = cap.read()
        if not hasFrame:
            logging.info("End of video or unable to capture frame.")
            break

        # Detect faces
        frameFace, bboxes = getFaceBox(faceNet, frame, conf_threshold=args.conf_threshold)
        if not bboxes:
            logging.info("No face detected in this frame.")
            continue

        # Loop through detected faces
        for bbox in bboxes:
            face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                         max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

            # Predict gender
            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            logging.info(f"Gender: {gender}, confidence = {genderPreds[0].max():.3f}")

            # Predict age
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            logging.info(f"Age: {age}, confidence = {agePreds[0].max():.3f}")

            # Display gender and age
            label = f"{gender}, {age}"
            cv.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        
        # Display the frame with detection
        cv.imshow("Age Gender Recognition", frameFace)

        # Write to output if specified
        if args.output and output_writer:
            output_writer.write(frameFace)

    # Release the video capture and writer
    cap.release()
    if args.output and output_writer:
        output_writer.release()
    cv.destroyAllWindows()

    logging.info("Script completed successfully.")

if __name__ == "__main__":
    main()
