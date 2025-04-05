import argparse
import cv2
import numpy as np
import tensorflow as tf
import logging
from logging import getLogger

# Load colors for visualization
colors = np.random.uniform(0, 255, size=(80, 3))  
FORMAT = '%(asctime)s [%(levelname)s]: %(message)s'

logging.basicConfig(format=FORMAT)

logger = getLogger(__name__)

def set_logging(level='INFO'):
    """Set logging level"""
    logger.setLevel(level)
    logger.info(f"Logging level set to {level}")
    
set_logging()

def load_labels(path):
    """Load labels from text file (one label per line)"""
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def main(tflite_model, labels_path):
    """
    Main function to load TFLite model, perform inference, and draw bounding boxes
    """
    # Load labels
    labels = load_labels(labels_path)
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_model)
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    image_width, image_height = (640, 640)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        logger.warning("Cannot open camera")
        exit()
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            logger.warning("Can't receive frame (stream end?). Exiting ...")
            break
        
        resized_image= cv2.resize(frame, (640, 640))
        input_image = np.array(resized_image) #
        input_image = np.true_divide(input_image, 255, dtype=np.float32) 
        input_image = input_image[np.newaxis, :]
        
        interpreter.set_tensor(input_details[0]['index'], input_image)
        interpreter.invoke()
        
        outputs = interpreter.get_tensor(output_details[0]['index'])
        
        outputs = np.squeeze(outputs).T  # Transpose to [boxes, 4+1+classes]
        
        boxes_xywh = outputs[:, :4]
        scores = np.max(outputs[:, 4:], axis=1)
        classes = np.argmax(outputs[:, 4:], axis=1)
        
        indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, 0.25, 0.45)

        threshold = 0.25
        
        for i in indices:
            if scores[i] < threshold:
                continue
            
            x_center, y_center, width, height = boxes_xywh[i]
            
            x1 = int((x_center - width / 2) * image_width)
            y1 = int((y_center - height / 2) * image_height)
            x2 = int((x_center + width / 2) * image_width)
            y2 = int((y_center + height / 2) * image_height)
            
            image_with_box = cv2.rectangle(frame, (x1,y1), (x2,y2), [0,0,255], 1)
            image_with_box = cv2.putText(image_with_box, f"{labels[classes[i]]} {scores[i]:.2%}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 1)
            
        try:
            cv2.imshow('Detection Results', image_with_box)
        except:
            logger.warning("NO OBJECTS DETECTED")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
    cap.release()
    cv2.destroyAllWindows()    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n_float32.tflite", help="Input your TFLite model.")
    parser.add_argument("--labels", default="coco_labels.txt", help="Path to labels text file.")
    args = parser.parse_args()
    main(args.model, args.labels)
