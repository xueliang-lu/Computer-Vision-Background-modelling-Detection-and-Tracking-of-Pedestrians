# CSCI935/CSCI435 (S223) Computer Vision Algorithms and Systems
# Assignment 3
# Student ID: 8097471
# UOW login name: xl340

"""
Solutions to the tracking and selection of up to three (3) most close pedestrians:

1. Detection:
    Extract bounding boxes for new pedestrians in the frame using get_new_bboxes with DNN module and the pre-trained MobileNet SSD.
    Filter out bounding boxes based on confidence and size criteria.

2. Tracking:
    For each detected pedestrian, check for overlaps with current trackers using the track_draw function.
    Initialize new KCF trackers for pedestrians.
    Update existing trackers and render their bounding boxes on the frame.
    Remove or discard inactive or out-of-frame trackers.

3. Top 3 Close Pedestrians to the Camera:
    Sort pedestrians based on their positions in the frame using the pick_close_up function.
    Working on the hypothesis that pedestrians nearer to the frame's bottom are in closer to camera proximity in actual space.
    Highlight the top three closest pedestrians with bounding boxes.

"""

import cv2
import numpy as np
import sys

def initialize_video(video_file):
    # Opens and initializes the video for processing.
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Cannot open video.")
        exit()
    
    return cap

def frame_resize(frame):
    # Resizes the frame camparable to VGA resolution while maintaining its original aspect ratio for effeciency.
    h, w = frame.shape[:2]
    vga_w, vga_h = 640, 480
    scale_w = w / vga_w
    scale_h = h / vga_h
    scale_factor = min(scale_w, scale_h)
    new_width = int(w / scale_factor)
    new_height = int(h / scale_factor)
    resized_frame = cv2.resize(frame, (new_width, new_height))
    
    return resized_frame

def frame_combine(topleft, topright, bottomleft, bottomright):
    # Combines four frames into a single frame, placing them in a 2x2 grid.
    top = cv2.hconcat([topleft, topright])
    bottom = cv2.hconcat([bottomleft, bottomright])
    combined_frames = cv2.vconcat([top, bottom])
    
    return combined_frames

def remove_noises(img):
    # Applies morphological operations to the image to remove noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 
    
    return opening

def compute_iou(boxA, boxB):
    # Computes the Intersection over Union (IoU) between two bounding boxes.
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

def detect_objects(frame, fg_mask_rn):
    # Detects and classifies objects in a frame using connected components analysis.
    # Create a whole black background and merge component mask with it.
    connected_output = np.zeros((fg_mask_rn.shape[0], fg_mask_rn.shape[1], 3), dtype=np.uint8)
    num_labels, labels_im = cv2.connectedComponents(fg_mask_rn,connectivity=8)
    
    person_count, car_count, other_count = 0, 0, 0

    for i in range(1, num_labels):
        component_mask = (labels_im == i).astype(np.uint8) * 255
        if np.count_nonzero(component_mask) < 400:  
            continue
        component = cv2.bitwise_and(frame, frame, mask=component_mask)
        connected_output = cv2.bitwise_or(connected_output, component)
        # use aspect_ratio difference of width and height to seperate person, car and other objects.
        x, y, w, h = cv2.boundingRect(component_mask)
        aspect_ratio = float(w) / h
        if 0.2 < aspect_ratio <= 1:
            person_count += 1
        elif aspect_ratio > 1.1:
            car_count += 1
        else:
            other_count += 1

    return person_count, car_count, other_count, connected_output


def process_frame(frame, bg_subtractor):
    # Processes the frame: resizes and applies background subtraction.
    # Get 3 channels foreground mask, get background and removes noises.
    frame_resized = frame_resize(frame)
    fg_mask = bg_subtractor.apply(frame_resized)
    fg_mask_rn = remove_noises(fg_mask)
    fg_mask_color = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    background = bg_subtractor.getBackgroundImage()
    
    return frame_resized, fg_mask_rn, fg_mask_color, background

def load_model():
    # Loads the the pre-trained MobileNet SSD
    with open('object_detection_classes_coco.txt', 'r') as f:
        class_names = f.read().split('\n')
    model = cv2.dnn.readNet(model='frozen_inference_graph.pb', 
                            config='ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
                            framework='TensorFlow')
    
    return model

def get_new_bboxes(frame, output):
    # Extracts new bounding boxes for detected person in the frame.
    new_bboxes = []  
    overlapped_frame = frame.copy()
    image_height, image_width, _ = frame.shape
    MAX_AREA = 0.1 * image_width * image_height
    CONFIDENCE_LEVEL = 0.3 # 30% confidence level
    
    for detection in output[0, 0, :, :]:
        confidence = detection[2]
        class_id = detection[1]

        if confidence > CONFIDENCE_LEVEL and int(class_id) == 1:  # person class
            box_x = detection[3] * image_width
            box_y = detection[4] * image_height
            box_width = detection[5] * image_width - box_x
            box_height = detection[6] * image_height - box_y
            area = box_width * box_height
            
            if area <= MAX_AREA:
                new_bbox = (int(box_x), int(box_y), int(box_width), int(box_height))
                new_bboxes.append(new_bbox)
                cv2.rectangle(overlapped_frame, 
                              (int(box_x), int(box_y)), 
                              (int(box_x + box_width), int(box_y + box_height)), 
                              (0,0,255), 
                              thickness=2)
    
    return new_bboxes, overlapped_frame

def track_draw(frame,new_bboxes,tracked_objects,person_counter):
    # Tracks and draws bounding boxes around detected objects in the frame.
    DETECTION_INTERVAL = 10 
    updated_objects = []
    labelled_frame = frame.copy()
    image_height, image_width, _ = frame.shape

    for new_bbox in new_bboxes:
        if not any([compute_iou(new_bbox, obj['bbox']) > 0.1 for obj in tracked_objects]):
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, new_bbox)
            tracked_objects.append({
                'tracker': tracker,
                'bbox': new_bbox,
                'id': f"Person{person_counter}",
                'age': 0
            })
            person_counter += 1  

    for obj in tracked_objects:
        tracker = obj['tracker']
        ok, bbox = tracker.update(frame)
        if ok:
            obj['age'] = 0
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(labelled_frame, p1, p2, (0, 255, 0), 2, 1)
            # Draw the unique ID on top of the bounding box
            cv2.putText(labelled_frame, obj['id'], (int(bbox[0]), int(bbox[1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            obj['bbox'] = bbox
            updated_objects.append(obj)
        else:
            obj['age'] += 1 

    tracked_objects = updated_objects
    # check if the tracked objects are out of frame or have not been detected for a certain interval and filter them out.
    tracked_objects = [obj for obj in tracked_objects 
                       if obj['age'] < DETECTION_INTERVAL 
                       and obj['bbox'][0] >= 0 
                       and obj['bbox'][1] >= 0 
                       and obj['bbox'][0] + obj['bbox'][2] <= image_width 
                       and obj['bbox'][1] + obj['bbox'][3] <= image_height]
    
    return tracked_objects,labelled_frame, person_counter

def pick_close_up(frame,tracked_objects):
    # Highlights top 3 persons from the tracked objects for a closer view.
    close_up_frame = frame.copy()
    K = 3 # set the top k persons that close to camera
    sorted_tracked = sorted(tracked_objects, key=lambda x: x['bbox'][1] + x['bbox'][3], reverse=True)
    
    for obj in sorted_tracked[:K]:
        bbox = obj['bbox']
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(close_up_frame, p1, p2, (255, 0, 0), 2, 1)
    
    return close_up_frame

def task1(video_file):
    # Processes the video file using Gaussian Mixture background modelling for background subtraction to detect and count objects.
    cap= initialize_video(video_file)
    #bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=46, detectShadows=True)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    frame_count = 0
    
    while True:
        ret, o_frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame, fg_mask, fg_mask_color, background = process_frame(o_frame, bg_subtractor)
        person_count, car_count, other_count, connected_output = detect_objects(frame, fg_mask)
        
        total_objects = person_count + car_count + other_count
        print(f"Frame {frame_count:04}: {total_objects} objects ({person_count} persons, {car_count} cars, {other_count} others)") 
        
        combined_frames = frame_combine(frame, background, fg_mask_color, connected_output)
        cv2.imshow('Frame', combined_frames)
        
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def task2(video_file):
    # Processes the video file using a pre-trained object detection model to detect, track, and label person objects. 
    cap = initialize_video(video_file)
    model =load_model()
    person_counter = 1
    tracked_objects = []
    
    while True:
        ret, o_frame = cap.read()
        if not ret:
            break 
        frame = frame_resize(o_frame)

        blob = cv2.dnn.blobFromImage(image=frame, size=(300, 300), mean=(104, 117, 123), swapRB=True)
        model.setInput(blob)
        output = model.forward()  

        new_bboxes, overlapped_frame = get_new_bboxes(frame, output)
        tracked_objects,labelled_frame, person_counter = track_draw(frame,new_bboxes,tracked_objects,person_counter)
        close_up_frame = pick_close_up(frame,tracked_objects)
        combined_frames = frame_combine(frame, overlapped_frame, labelled_frame, close_up_frame)      
        cv2.imshow('Frame',combined_frames)
        
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def parse_and_run():
    # Parse command line arguments to determine which task to run.
    # wrong input handle
    if len(sys.argv)!= 3:
        print("Wrong input!")
        print("Please use correct format: –b (or -d) video_file_path.")
        return
    # input -b to execute task1() or -d to execute task2() and handle error input with prompting.
    else:
        if sys.argv[1]== "-b":
            task1(sys.argv[2])
        elif sys.argv[1]== "-d":
            task2(sys.argv[2])
        else:
            print("Please input first argument with '-b' for task 1 or '-d' for task 2.")
      
if __name__== '__main__':
    parse_and_run()