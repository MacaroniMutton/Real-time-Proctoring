import cv2
import imutils
import time
import threading
import queue
import numpy as np
import winsound
from datetime import datetime
import os
import dlib
import face_recognition  # For ID card face comparison

# Global variables
global data_record
data_record = []
processing_queue = queue.Queue(maxsize=10)  # Buffer for frames
results_queue = queue.Queue()  # Queue for detection results
detection_active = True  # Flag to control detection threads
reference_image = None  # Will store the ID card face image

# For Beeping
frequency = 2500
duration = 1000

# Preload models to avoid loading during runtime
print("Loading models...")
# Face detection
face_detector = dlib.get_frontal_face_detector()

# Load shape predictor only once
shape_predictor_path = 'shape_predictor_model/shape_predictor_68_face_landmarks.dat'
if not os.path.exists(shape_predictor_path):
    print(f"ERROR: Shape predictor model not found at {shape_predictor_path}")
    exit(1)

shape_predictor = dlib.shape_predictor(shape_predictor_path)

# Load YOLO model
yolo_weights = "object_detection_model/weights/yolov3-tiny.weights"
yolo_config = "object_detection_model/config/yolov3-tiny.cfg"

if not (os.path.exists(yolo_weights) and os.path.exists(yolo_config)):
    print(f"ERROR: YOLO model files not found at {yolo_weights} or {yolo_config}")
    exit(1)

net = cv2.dnn.readNet(yolo_weights, yolo_config)

# Load COCO class names
label_classes = []
with open("object_detection_model/objectLabels/coco.names", "r") as file:
    label_classes = [name.strip() for name in file.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

# 3D Model points for head pose estimation
model_points = np.array([
    (0.0, 0.0, 0.0),            # Nose tip
    (0.0, -330.0, -65.0),       # Chin
    (-255.0, 170.0, -135.0),    # Left eye left corner
    (225.0, 170.0, -135.0),     # Right eye right corner
    (-150.0, -150.0, -125.0),   # Left mouth corner
    (150.0, -150.0, -125.0)     # Right mouth corner
])

print("Models loaded successfully!")

# Performance optimization settings
SKIP_FRAMES = 5  # Process only every Nth frame for heavy operations
YOLO_SKIP_FRAMES = 15  # YOLO is even heavier, skip more frames
MAX_WIDTH = 640  # Resize frames to this width for faster processing
DETECTION_INTERVAL = 0.1  # Time between detections in seconds

# ID card verification settings
ID_VERIFICATION_INTERVAL = 100  # Check identity every N frames
FACE_MATCH_THRESHOLD = 0.6  # Lower value = stricter matching

# Camera settings
def init_camera():
    cam = cv2.VideoCapture(0)
    
    # Set lower resolution for better performance
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Try to set higher FPS if camera supports it
    cam.set(cv2.CAP_PROP_FPS, 30)
    
    # Additional settings to reduce latency
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cam.isOpened():
        print("Error: Could not open camera.")
        return None
    
    return cam

# --- ID CARD VERIFICATION FUNCTIONS ---

def capture_id_card():
    """Capture and process ID card image"""
    print("\n=========================================")
    print("ID CARD VERIFICATION")
    print("=========================================")
    print("Please show your ID card to the camera.")
    print("Make sure your face on the ID is clearly visible.")
    print("Press 'c' to capture your ID card when ready.")
    print("=========================================\n")
    
    cam = init_camera()
    if cam is None:
        print("Failed to initialize camera for ID verification")
        return None
        
    id_captured = False
    id_image = None
    
    try:
        while not id_captured:
            ret, frame = cam.read()
            
            if not ret:
                print("Failed to grab frame")
                time.sleep(0.1)
                continue
                
            # Resize frame for better performance
            frame = imutils.resize(frame, width=MAX_WIDTH)
            
            # Display instructions
            cv2.putText(frame, "Show ID card and press 'c' to capture", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                       
            # Display frame
            cv2.imshow('ID Card Verification', frame)
            
            # Check for capture key or exit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                id_image = frame.copy()
                id_captured = True
                print("ID card captured successfully!")
                
                # Save ID card image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                id_filename = f"id_card_{timestamp}.jpg"
                cv2.imwrite(id_filename, id_image)
                print(f"ID card image saved as {id_filename}")
                
                # Extract face from ID card
                id_face_encoding = process_id_card(id_image)
                if id_face_encoding is not None:
                    print("Face detected on ID card!")
                else:
                    print("WARNING: No face detected on ID card. Please try again.")
                    id_captured = False
                    
            elif key == ord('q'):
                print("ID verification cancelled")
                return None
                
    finally:
        # Clean up
        cam.release()
        cv2.destroyAllWindows()
        
    return id_face_encoding

def process_id_card(id_image):
    """Extract face encoding from ID card image"""
    # Find faces in the ID card image
    face_locations = face_recognition.face_locations(id_image)
    
    if len(face_locations) == 0:
        print("No face found on ID card")
        return None
        
    if len(face_locations) > 1:
        print(f"Multiple faces ({len(face_locations)}) found on ID card. Using the first one.")
        
    # Get face encoding of the first face
    face_encodings = face_recognition.face_encodings(id_image, face_locations)
    
    if not face_encodings:
        return None
        
    return face_encodings[0]

def verify_identity(frame, reference_encoding):
    """Verify if the person in frame matches the ID card"""
    if reference_encoding is None:
        return False, "No reference face available"
        
    # Find faces in current frame
    face_locations = face_recognition.face_locations(frame)
    
    if len(face_locations) == 0:
        return False, "No face detected in current frame"
        
    if len(face_locations) > 1:
        return False, f"Multiple faces ({len(face_locations)}) detected"
        
    # Get face encoding of the first face
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    if not face_encodings:
        return False, "Could not encode detected face"
        
    # Compare with reference face
    match = face_recognition.compare_faces([reference_encoding], face_encodings[0], 
                                          tolerance=FACE_MATCH_THRESHOLD)
    
    distance = face_recognition.face_distance([reference_encoding], face_encodings[0])[0]
    
    if match[0]:
        confidence = (1.0 - distance) * 100
        return True, f"Identity verified (confidence: {confidence:.1f}%)"
    else:
        return False, f"Identity NOT verified (distance: {distance:.3f})"

# --- DETECTION FUNCTIONS (OPTIMIZED) ---

def detect_face(frame):
    """
    Optimized face detection function
    Returns the count of faces and face objects detected by dlib
    """
    # Convert to grayscale once
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_detector(gray, 0)
    face_count = len(faces)
    
    # Minimal drawing for better performance
    if face_count > 0:
        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            # Draw simple rectangle instead of fancy corners
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
    
    return face_count, faces, gray

def is_blinking(faces, frame, gray):
    """Optimized blink detection"""
    for face in faces:
        facial_landmarks = shape_predictor(gray, face)
        
        # Left eye
        left_eye_ratio = calculate_eye_ratio(facial_landmarks, 36, 37, 38, 39, 40, 41)
        
        # Right eye
        right_eye_ratio = calculate_eye_ratio(facial_landmarks, 42, 43, 44, 45, 46, 47)
        
        # Check for blink
        if left_eye_ratio >= 3.6 or right_eye_ratio >= 3.6:
            return True, "Blink"
        
    return False, "No Blink"

def calculate_eye_ratio(landmarks, p1, p2, p3, p4, p5, p6):
    """Helper function to calculate eye aspect ratio"""
    # Calculate horizontal length
    left_point = (landmarks.part(p1).x, landmarks.part(p1).y)
    right_point = (landmarks.part(p4).x, landmarks.part(p4).y)
    hor_len = calc_distance(left_point, right_point)
    
    # Calculate vertical length - average of two distances
    top_mid = mid_point(landmarks.part(p2), landmarks.part(p3))
    bottom_mid = mid_point(landmarks.part(p5), landmarks.part(p6))
    ver_len = calc_distance(top_mid, bottom_mid)
    
    # Avoid division by zero
    if ver_len == 0:
        return 0
        
    # Calculate ratio
    return hor_len / ver_len

def mid_point(point1, point2):
    """Calculate midpoint between two points"""
    x = (point1.x + point2.x) // 2
    y = (point1.y + point2.y) // 2
    return (x, y)

def calc_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    from math import hypot
    return hypot(point1[0] - point2[0], point1[1] - point2[1])

def detect_gaze(faces, frame, gray):
    """Optimized gaze detection"""
    for face in faces:
        facial_landmarks = shape_predictor(gray, face)
        
        # Extract eye regions
        left_eye = extract_eye_region(facial_landmarks, [36, 37, 38, 39, 40, 41], frame, gray)
        right_eye = extract_eye_region(facial_landmarks, [42, 43, 44, 45, 46, 47], frame, gray)
        
        if left_eye is None or right_eye is None:
            return "center"  # Default if eyes cannot be processed
            
        # Process left eye
        left_gaze = process_eye_gaze(left_eye)
        
        # Process right eye
        right_gaze = process_eye_gaze(right_eye)
        
        # Determine overall gaze direction
        if left_gaze == "right" and right_gaze == "right":
            return "right"
        elif left_gaze == "left" and right_gaze == "left":
            return "left"
        else:
            return "center"
    
    return "center"  # Default return value

def extract_eye_region(landmarks, eye_points, frame, gray):
    """Extract eye region for gaze detection"""
    try:
        # Get region points
        region = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_points], np.int32)
        
        # Get region boundaries
        min_x = np.min(region[:,0])
        max_x = np.max(region[:,0])
        min_y = np.min(region[:,1])
        max_y = np.max(region[:,1])
        
        # Create mask and extract eye
        height, width = gray.shape
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [region], 255)
        eye = cv2.bitwise_and(gray, gray, mask=mask)
        
        # Crop to eye region
        eye = eye[min_y:max_y, min_x:max_x]
        
        if eye.size == 0:
            return None
            
        # Apply threshold
        _, eye = cv2.threshold(eye, 70, 255, cv2.THRESH_BINARY)
        return eye
    except Exception:
        return None

def process_eye_gaze(eye):
    """Process eye to determine gaze direction"""
    if eye is None or eye.size == 0:
        return "center"
        
    height, width = eye.shape
    
    # Skip if eye is too small
    if width < 4:
        return "center"
        
    # Divide eye into left and right portions
    left_side = eye[0:height, 0:width//2]
    right_side = eye[0:height, width//2:width]
    
    # Count white pixels
    left_white = cv2.countNonZero(left_side)
    right_white = cv2.countNonZero(right_side)
    
    # Avoid division by zero
    if left_white == 0:
        left_white = 1
    if right_white == 0:
        right_white = 1
        
    # Determine gaze direction
    ratio_threshold = 1.5
    if right_white / left_white > ratio_threshold:
        return "right"
    elif left_white / right_white > ratio_threshold:
        return "left"
    else:
        return "center"

def track_mouth(faces, frame, gray):
    """Optimized mouth tracking"""
    for face in faces:
        facial_landmarks = shape_predictor(gray, face)
        
        # Get top and bottom lip points
        top_lip = facial_landmarks.part(51)
        bottom_lip = facial_landmarks.part(57)
        
        # Calculate distance
        distance = calc_distance((top_lip.x, top_lip.y), (bottom_lip.x, bottom_lip.y))
        
        # Threshold for mouth open
        if distance > 20:
            return "Mouth Open"
        else:
            return "Mouth Closed"
    
    return "No Mouth Detected"

def detect_object(frame, frame_count):
    """
    Optimized YOLO-based object detection
    Only run this on every YOLO_SKIP_FRAMES frame
    """
    # Skip frames for better performance
    if frame_count % YOLO_SKIP_FRAMES != 0:
        return []
        
    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (320, 240))
    height, width, _ = small_frame.shape
    
    # Prepare blob
    blob = cv2.dnn.blobFromImage(small_frame, 0.00392, (224, 224), (0, 0, 0), True, crop=False)
    
    # Forward pass through network
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Process results
    class_ids = []
    confidences = []
    boxes = []
    detected_objects = []
    
    # Only consider objects with high probability
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.3:  # Higher threshold for better performance
                # Get box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Create result list
    for i in range(len(boxes)):
        if i in indexes:
            label = str(label_classes[class_ids[i]])
            confidence = confidences[i]
            detected_objects.append((label, confidence))
            
            # Draw on frame
            i = i.item() if hasattr(i, 'item') else i  # Handle numpy int64 if needed
            x, y, w, h = boxes[i]
            
            # Scale coordinates back to original frame
            scale_x = frame.shape[1] / small_frame.shape[1]
            scale_y = frame.shape[0] / small_frame.shape[0]
            
            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label_text = f"{label} {confidence:.2f}"
            cv2.putText(frame, label_text, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return detected_objects

def detect_head_pose(faces, frame, gray):
    """Optimized head pose detection"""
    for face in faces:
        marks = shape_predictor(gray, face)
        
        # Get 2D image points
        image_points = np.array([
            [marks.part(30).x, marks.part(30).y],    # Nose tip
            [marks.part(8).x, marks.part(8).y],      # Chin
            [marks.part(36).x, marks.part(36).y],    # Left eye left corner
            [marks.part(45).x, marks.part(45).y],    # Right eye right corner
            [marks.part(48).x, marks.part(48).y],    # Left Mouth corner
            [marks.part(54).x, marks.part(54).y]     # Right mouth corner
        ], dtype="double")
        
        # Camera matrix
        focal_length = frame.shape[1]
        center = (frame.shape[1]/2, frame.shape[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]], dtype="double"
        )
        
        # Solve for pose
        dist_coeffs = np.zeros((4,1))
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return "Unknown"
            
        # Project a 3D point for direction
        (nose_end_point2D, _) = cv2.projectPoints(
            np.array([(0.0, 0.0, 1000.0)]), 
            rotation_vector, translation_vector, 
            camera_matrix, dist_coeffs
        )
        
        # Calculate angles
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        
        try:
            # Vertical angle
            m = (p2[1] - p1[1]) / (p2[0] - p1[0]) if p2[0] != p1[0] else float('inf')
            ang1 = int(np.degrees(np.arctan(m)))
            
            # Horizontal angle - simplified calculation
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            ang2 = int(np.degrees(np.arctan2(dy, dx)))
            
            # Determine head pose
            if ang1 >= 30:
                return "Head Up"
            elif ang1 <= -30:
                return "Head Down"
            elif ang2 >= 30:
                return "Head Right"
            elif ang2 <= -30:
                return "Head Left"
            else:
                return "Head Center"
                
        except:
            return "Unknown"
    
    return "No Face"

# --- WORKER THREADS ---

def camera_thread():
    """Thread to capture frames from the camera"""
    global detection_active
    
    # Initialize camera
    cam = init_camera()
    if cam is None:
        print("Failed to initialize camera")
        detection_active = False
        return
        
    frame_count = 0
    last_frame_time = time.time()
    fps = 0
    
    try:
        while detection_active:
            # Capture frame
            ret, frame = cam.read()
            
            if not ret:
                print("Failed to grab frame")
                time.sleep(0.1)
                continue
                
            # Resize frame for better performance
            frame = imutils.resize(frame, width=MAX_WIDTH)
            
            # Calculate and display FPS
            current_time = time.time()
            elapsed = current_time - last_frame_time
            
            if elapsed > 0:
                fps = 1 / elapsed
                last_frame_time = current_time
                
            # Display FPS info
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Put frame in queue for processing
            if not processing_queue.full():
                processing_queue.put((frame.copy(), frame_count))
                
            # Show frame
            cv2.imshow('Proctoring System', frame)
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                detection_active = False
                break
                
            frame_count += 1
                
    finally:
        # Clean up
        cam.release()
        cv2.destroyAllWindows()

def detection_thread():
    """Thread to process frames for detections"""
    global detection_active, reference_image
    
    frame_count = 0
    blink_count = 0
    last_detected_objects = []
    last_face_count = 0
    last_head_pose = "Unknown"
    last_mouth_status = "Unknown"
    last_gaze = "center"
    last_id_check_frame = 0
    last_id_match_status = "Not checked yet"
    
    while detection_active:
        try:
            # Get frame from queue with timeout
            frame, count = processing_queue.get(timeout=1.0)
            frame_count = count
            
            # Start timing the processing
            start_time = time.time()
            
            # Record current time
            current_time = datetime.now().strftime("%H:%M:%S.%f")
            
            # Initialize record
            record = [current_time]
            
            # Run face detection on every frame
            face_count, faces, gray = detect_face(frame)
            
            # Save previous face count
            last_face_count = face_count
            
            # Process face count
            face_status = "Face detecting properly."
            if face_count > 1:
                face_status = "Multiple faces have been detected."
            elif face_count == 0:
                face_status = "No face has been detected."
                
            record.append(face_status)
            
            # Only process further if at least one face is detected
            if face_count == 1:
                # 1. Blink detection - process more often
                if frame_count % 3 == 0:
                    is_blinking_result, blink_status = is_blinking(faces, frame, gray)
                    if is_blinking_result:
                        blink_count += 1
                        record.append(f"Blink count: {blink_count}")
                    else:
                        record.append(blink_status)
                else:
                    record.append("Blink detection skipped")
                
                # 2. Gaze detection - process every few frames
                if frame_count % SKIP_FRAMES == 0:
                    last_gaze = detect_gaze(faces, frame, gray)
                record.append(last_gaze)
                
                # 3. Mouth tracking - process every few frames
                if frame_count % SKIP_FRAMES == 0:
                    last_mouth_status = track_mouth(faces, frame, gray)
                record.append(last_mouth_status)
                
                # 4. Object detection - process less frequently
                if frame_count % YOLO_SKIP_FRAMES == 0:
                    last_detected_objects = detect_object(frame, frame_count)
                
                record.append(str(last_detected_objects))
                
                # Warning for detected objects
                if len(last_detected_objects) > 1:
                    # Trigger warning in separate thread to avoid blocking
                    thread = threading.Thread(target=lambda: winsound.Beep(frequency, duration))
                    thread.daemon = True
                    thread.start()
                
                # 5. Head pose detection - process every few frames
                if frame_count % SKIP_FRAMES == 0:
                    last_head_pose = detect_head_pose(faces, frame, gray)
                record.append(last_head_pose)
                
                # 6. ID verification - check periodically
                if reference_image is not None and frame_count - last_id_check_frame >= ID_VERIFICATION_INTERVAL:
                    last_id_check_frame = frame_count
                    id_match, id_status = verify_identity(frame, reference_image)
                    last_id_match_status = id_status
                    
                    # If identity doesn't match, sound alarm
                    if not id_match:
                        thread = threading.Thread(target=lambda: winsound.Beep(frequency, duration))
                        thread.daemon = True
                        thread.start()
                
                record.append(last_id_match_status)
                
            # Add record to data_record
            data_record.append(record)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Add results to results queue
            results_queue.put({
                'face_count': face_count,
                'blink_count': blink_count,
                'gaze': last_gaze,
                'mouth': last_mouth_status,
                'objects': last_detected_objects,
                'head_pose': last_head_pose,
                'identity': last_id_match_status,
                'processing_time': processing_time
            })
            
            # Indicate we're done with this frame
            processing_queue.task_done()
            
        except queue.Empty:
            # No frames available, just continue
            continue
        except Exception as e:
            print(f"Error in detection thread: {e}")
            # Add minimal record to prevent data inconsistency
            data_record.append([datetime.now().strftime("%H:%M:%S.%f"), "Error processing frame"])
            
    print("Detection thread stopped")

# --- MAIN FUNCTION ---

def proctoring_algo():
    """Main proctoring algorithm with threading"""
    global detection_active, reference_image
    
    try:
        print("Starting proctoring system...")
        
        # First, capture ID card
        print("Starting ID card verification...")
        reference_image = capture_id_card()
        
        if reference_image is None:
            print("WARNING: No ID card face captured. Identity verification disabled.")
        else:
            print("ID card face captured successfully. Identity verification enabled.")
        
        print("\nStarting main proctoring system...")
        print("Press 'q' to quit")
        
        # Start camera thread
        cam_thread = threading.Thread(target=camera_thread)
        cam_thread.daemon = True
        cam_thread.start()
        
        # Start detection thread
        detect_thread = threading.Thread(target=detection_thread)
        detect_thread.daemon = True
        detect_thread.start()
        
        # Main thread now just monitors and handles user input
        while detection_active:
            time.sleep(0.1)
            
            # Display all results from results queue
            try:
                while not results_queue.empty():
                    result = results_queue.get(False)
                    
                    # Print a separator for better readability
                    print("\n" + "="*60)
                    print(f"DETECTION RESULTS ({datetime.now().strftime('%H:%M:%S')})")
                    print("="*60)
                    
                    # Print all detection results
                    print(f"Processing time: {result.get('processing_time', 0)*1000:.1f}ms")
                    print(f"Face count: {result.get('face_count', 0)}")
                    print(f"Blink count: {result.get('blink_count', 0)}")
                    print(f"Eye gaze: {result.get('gaze', 'Unknown')}")
                    print(f"Mouth status: {result.get('mouth', 'Unknown')}")
                    print(f"Head pose: {result.get('head_pose', 'Unknown')}")
                    
                    # Print detected objects with confidence scores
                    objects = result.get('objects', [])
                    if objects:
                        print(f"Detected objects ({len(objects)}):")
                        for obj, conf in objects:
                            print(f"  - {obj} (confidence: {conf:.2f})")
                    else:
                        print("  - No objects detected")
                    
                    # Print identity verification status
                    print(f"Identity verification: {result.get('identity', 'Not checked')}")
                    print("="*60)
                    
                    results_queue.task_done()
            except queue.Empty:
                pass
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        detection_active = False
        
        # Wait for threads to finish
        if 'cam_thread' in locals() and cam_thread.is_alive():
            cam_thread.join(timeout=1.0)
            
        if 'detect_thread' in locals() and detect_thread.is_alive():
            detect_thread.join(timeout=1.0)
            
        # Save activity data
        print("Saving activity data...")
        activity_val = "\n".join(map(str, data_record))
        
        with open('activity.txt', 'w') as file:
            file.write(str(activity_val))
            
        print("Proctoring system stopped")

if __name__ == '__main__':
    proctoring_algo()