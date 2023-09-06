
import cv2
import argparse
import numpy as np

class Yolo:
    def __init__(self, width, height, classes_path="./yolo/yolov3.txt", weights_path="./yolo/yolov3.weights", config_path="./yolo/yolov3.cfg") -> None:
        self.width = width
        self.height = height
        self.scale = 0.00392
        self.classes = None
        self.class_ids = []
        self.confidences = []
        self.boxes = []
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4
        with open(classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.net = cv2.dnn.readNet(weights_path, config_path)

    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    def draw_bounding_box(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(self.classes[class_id])
        color = self.COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def detect_objects(self, image):
        self.class_ids = []
        self.confidences = []
        self.boxes = []
        self.blob = cv2.dnn.blobFromImage(image, self.scale, (416,416), (0,0,0), True, crop=False)
        self.net.setInput(self.blob)
        outs = self.net.forward(self.get_output_layers(self.net))
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0]*self.width)
                    center_y = int(detection[1]*self.height)
                    w = int(detection[2] * self.width)
                    h = int(detection[3] * self.height)
                    x = center_x - w/2
                    y = center_y - h/2
                    self.class_ids.append(class_id)
                    self.confidences.append(float(confidence))
                    self.boxes.append([x,y,w,h])
    
    def draw_detection_box(self, image):
        indices = cv2.dnn.NMSBoxes(self.boxes, self.confidences, self.conf_threshold, self.nms_threshold)
        for i in indices:
            i = i
            box = self.boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            self.draw_bounding_box(image, self.class_ids[i], self.confidences[i], round(x), round(y), round(x+w), round(y+h))

