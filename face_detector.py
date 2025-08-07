import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils

class FaceDetector:
    def __init__(self):

        # dlib's face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        
        
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #hugging face
        
        # Facial Landmarks Map
        #- Jaw: points 0-16
        #- Right eyebrow: points 17-21
        #- Left eyebrow: points 22-26
        #- Nose bridge: points 27-30
        #- Lower nose: points 31-35
        #- Right eye: points 36-41
        #- Left eye: points 42-47
        #- Outer lip: points 48-59
        #- Inner lip: points 60-67

        # facial landmark indexes for eyes
        self.LEFT_EYE_START = 36
        self.LEFT_EYE_END = 42
        self.RIGHT_EYE_START = 42
        self.RIGHT_EYE_END = 48
        
        # nose tip index
        self.NOSE_TIP = 30
        
        
        # head position estimation points, for pnp
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip, ref point
            (0.0, -330.0, -65.0),        # Chin, straight down slightly back
            (-225.0, 170.0, -135.0),     # Left eye left corner, > left, up, back (close open)
            (225.0, 170.0, -135.0),      # Right eye right corner > right up, back 
            (-150.0, -150.0, -125.0),    # Left Mouth corner > left, down, back
            (150.0, -150.0, -125.0)      # Right mouth corner > right, down, back
        ])
        
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray) #green boundary, using dlib
        return faces, gray
    
    def get_landmarks(self, gray, face):

        landmarks = self.predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        return landmarks
    
    #vertical n horizontal distance between eye n landmark
    def calculate_ear(self, eye): #Eye Aspect Ratio (EAR)

        #using euclidean distances between the:

        # two sets of vertical eye, x and y coordinates, vertical
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        
        # between the horizontal eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])
        
        # eye aspect ratio, larger> eye open > higher ear value
        ear = (A + B) / (2.0 * C)
        return ear
    
    #BASED ON EAR
    def is_blinking(self, landmarks):

        # extract eye coordinates
        left_eye = landmarks[self.LEFT_EYE_START:self.LEFT_EYE_END]
        right_eye = landmarks[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
        
        #calculate EAR for both eyes
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        
        #average
        ear = (left_ear + right_ear) / 2.0
        
        #threshold for ear
        return ear < 0.25, ear
    
    #head position - 
    #give the angles 
    def get_head_pose(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        
        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        
        dist_coeffs = np.zeros((4, 1))
        
        image_points = np.array([
            landmarks[30],     # Nose tip
            landmarks[8],      # Chin
            landmarks[36],     # Left eye left corner
            landmarks[45],     # Right eye right corner
            landmarks[48],     # Left mouth corner
            landmarks[54]      # Right mouth corner
        ], dtype="double")
        


        # Solve PnP - perspective-n-point
        #estimating the pose (position and orientation)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs
        )
        
        #raw rotation data from the camera

        if success:
            #rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            #Euler angles
            sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
            singular = sy < 1e-6


            if not singular:#for normal head position, covnert matrix into angles
                x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2]) #pitch
                y = np.arctan2(-rotation_matrix[2,0], sy) #left right, yaw
                z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0]) #tilting side to side
            else: #exterme angles
                x = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
                y = np.arctan2(-rotation_matrix[2,0], sy)
                z = 0 #
            
            #degrees
            pitch = np.degrees(x) #updown
            yaw = np.degrees(y) #leftright
            roll = np.degrees(z) #tilt
            
            #something is tilted, turned, or rotated

            return pitch, yaw, roll
        
        return None, None, None
    
    #2D, give the direction
    def get_face_direction(self, landmarks):

        # nose tip and face boundaries
        nose_tip = landmarks[self.NOSE_TIP]
        
        # face bounding box from landmarks
        left_face = min(landmarks[:, 0])
        right_face = max(landmarks[:, 0])
        face_width = right_face - left_face
        face_center_x = left_face + face_width / 2
        
        #relative position of nose tip
        nose_relative = (nose_tip[0] - face_center_x) / (face_width / 2)
        
        # THRESHOLD for direction detection
        if nose_relative < -0.15:
            return "left"
        elif nose_relative > 0.15:
            return "right"
        else:
            return "center"
    
    def draw_landmarks(self, frame, landmarks):
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
        return frame
    
    def draw_face_boundary(self, frame, face):
        
        #rectangle to the detected face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return frame
    
    #present of eyeglass using facial landmarks & edge detection
    #false positive, nagdedetetc ng eyeglasses kahit wala naman

    #def detect_eyeglasses(self, landmarks, frame):
    #
    #    # extract eye regions first
    #    left_eye = landmarks[self.LEFT_EYE_START:self.LEFT_EYE_END]
    #    right_eye = landmarks[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
    #    
    #    #bridge of nose area where glasses typically sit
    #    nose_bridge_points = landmarks[27:31]  # nose bridge landmarks
    #    
    #    def check_glasses_on_eye(eye_landmarks, frame):
    #        # eye region with smaller expansion to detect glasses frames
    #        x_coords = eye_landmarks[:, 0]
    #        y_coords = eye_landmarks[:, 1]
    #        
    #        # just around the eye
    #        center_x = int(np.mean(x_coords))
    #        center_y = int(np.mean(y_coords))
    #        
    #        #  possible region where glasses frames would be, adjust
    #        width = int((max(x_coords) - min(x_coords)) * 1.2)
    #        height = int((max(y_coords) - min(y_coords)) * 1.2)
    #        
    #        x1 = max(0, center_x - width//2)
    #        y1 = max(0, center_y - height//2 - 10) 
    #        x2 = min(frame.shape[1], center_x + width//2)
    #        y2 = min(frame.shape[0], center_y + height//2)
    #        
    #        eye_region = frame[y1:y2, x1:x2]
    #        
    #        if eye_region.size == 0:
    #            return False
    #            
    #        #since different eyegalsses frames has diff shades
    #        gray_region = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    #        
    #        thresh = cv2.adaptiveThreshold(gray_region, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    #        
    #        # detect horizontal lines for glasses frames
    #        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    #        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    #        
    #        #count horizontal structures
    #        horizontal_pixels = np.sum(horizontal_lines > 0)
    #        total_pixels = horizontal_lines.size
    #        
    #        # threshold for glasses detection
    #        return (horizontal_pixels / total_pixels) > 0.10
    #    
    #    # eyes and nose bridge area
    #    left_glasses = check_glasses_on_eye(left_eye, frame)
    #    right_glasses = check_glasses_on_eye(right_eye, frame)
    #    
    #    # nose bridge for glasses bridge
    #    nose_x_coords = nose_bridge_points[:, 0]
    #    nose_y_coords = nose_bridge_points[:, 1]
    #    
    #    bridge_x1 = max(0, int(min(nose_x_coords)) - 10)
    #    bridge_y1 = max(0, int(min(nose_y_coords)) - 5)
    #    bridge_x2 = min(frame.shape[1], int(max(nose_x_coords)) + 10)
    #    bridge_y2 = min(frame.shape[0], int(max(nose_y_coords)) + 5)
    #    
    #    bridge_region = frame[bridge_y1:bridge_y2, bridge_x1:bridge_x2]
    #    bridge_glasses = False
    #    
    #    if bridge_region.size > 0:
    #        gray_bridge = cv2.cvtColor(bridge_region, cv2.COLOR_BGR2GRAY)
    #
    #        # dark horizontal line across nose bridge
    #        bridge_thresh = cv2.threshold(gray_bridge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    #        dark_pixels = np.sum(bridge_thresh == 0)
    #        total_bridge_pixels = bridge_thresh.size
    #        bridge_glasses = (dark_pixels / total_bridge_pixels) > 0.3
    #    
    #    # glasses detected, found on both eyes OR nose bridge
    #    return (left_glasses and right_glasses) or bridge_glasses
    #

    #backup glassess detection, if consistent horizontal lines
    def glasses_detection(self, landmarks, frame):
        """Simple backup glasses detection focusing on consistent horizontal lines"""
        try:
            left_eye = landmarks[self.LEFT_EYE_START:self.LEFT_EYE_END]
            right_eye = landmarks[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
            
            # a horizontal strip across both eyes where glasses would be
            left_eye_y = int(np.mean(left_eye[:, 1]))
            right_eye_y = int(np.mean(right_eye[:, 1]))
            
            # Average Y position of both eyes
            glasses_y = (left_eye_y + right_eye_y) // 2
            
            # X coordinates spanning both eyes
            left_x = int(min(left_eye[:, 0])) - 20
            right_x = int(max(right_eye[:, 0])) + 20
            
            # extract horizontal strip
            strip_top = max(0, glasses_y - 8)
            strip_bottom = min(frame.shape[0], glasses_y + 8)
            strip_left = max(0, left_x)
            strip_right = min(frame.shape[1], right_x)
            
            glasses_strip = frame[strip_top:strip_bottom, strip_left:strip_right]
            
            if glasses_strip.size == 0:
                return False
            
            # to grayscale and look for consistent dark horizontal line
            gray_strip = cv2.cvtColor(glasses_strip, cv2.COLOR_BGR2GRAY)
            
            # threshold to find dark areas
            _, thresh = cv2.threshold(gray_strip, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Count dark pixels in each row
            dark_pixels_per_row = []
            for row in thresh:
                dark_count = np.sum(row == 0)  # Dark pixels
                dark_pixels_per_row.append(dark_count / len(row))
            
            # consistent dark horizontal line
            max_dark_ratio = max(dark_pixels_per_row) if dark_pixels_per_row else 0
            
            # likely glasses (less sensitive)
            return max_dark_ratio > 0.8
        
    
            
        except Exception:
            return False
    

    
    #face obstructions
    def check_face_coverage(self, landmarks, frame):
        """Check for face obstructions: glasses, bangs, masks, etc."""
        issues = []
        
        try:
            # for eyeglasses (include in face coverage)
            try:
                glasses_detected = self.glasses_detection(landmarks, frame)
                if glasses_detected:
                    issues.append("Eyeglasses detected")
            except:
                pass
            
            # forehead area
            eyebrow_points = landmarks[17:27]  # eyebrow
            if len(eyebrow_points) > 0:

                #forehead region
                forehead_top_y = int(min(eyebrow_points[:, 1])) - 40
                forehead_bottom_y = int(min(eyebrow_points[:, 1]))
                face_left = int(min(landmarks[:, 0]))
                face_right = int(max(landmarks[:, 0]))
                
                # check if valid forehead region
                if forehead_top_y >= 0 and forehead_bottom_y > forehead_top_y:
                    forehead_region = frame[forehead_top_y:forehead_bottom_y, face_left:face_right]
                    
                    if forehead_region.size > 0:
                        gray_forehead = cv2.cvtColor(forehead_region, cv2.COLOR_BGR2GRAY)
                        
                        # adaptive for light and dark color hair
                        mean_brightness = np.mean(gray_forehead)
                        
                        # for lighter hair, use higher threshold; for darker hair, use lower threshold
                        adaptive_threshold = max(50, min(80, int(mean_brightness * 0.7)))
                        _, thresh = cv2.threshold(gray_forehead, adaptive_threshold, 255, cv2.THRESH_BINARY)
                        
                        #  pixels that are darker than the adaptive threshold (hair/bangs)
                        hair_pixels = np.sum(thresh == 0)
                        total_pixels = thresh.size
                        hair_ratio = hair_pixels / total_pixels if total_pixels > 0 else 0
                        
                        # threshold for hair detection 
                        if hair_ratio > 0.60:
                            issues.append("Obstruction covering forehead")
            
            # objects covering face center
            face_center_x = int(np.mean(landmarks[:, 0]))
            face_center_y = int(np.mean(landmarks[:, 1]))
            
            # region for face obstruction
            center_size = 50
            center_left = max(0, face_center_x - center_size//2)
            center_right = min(frame.shape[1], face_center_x + center_size//2)
            center_top = max(0, face_center_y - center_size//2)
            center_bottom = min(frame.shape[0], face_center_y + center_size//2)
            
            center_region = frame[center_top:center_bottom, center_left:center_right]
            
            if center_region.size > 0:
                
                gray_center = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
                avg_brightness = np.mean(gray_center)
                
                if avg_brightness < 20:  #to reduce false positives
                    issues.append("Face center obstructed (mask/object)")
            
            # eye area for obstructions
            left_eye = landmarks[self.LEFT_EYE_START:self.LEFT_EYE_END]
            right_eye = landmarks[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
            
            def check_eye_obstruction(eye_landmarks, eye_name):
                eye_center_x = int(np.mean(eye_landmarks[:, 0]))
                eye_center_y = int(np.mean(eye_landmarks[:, 1]))
                
                # Check area around eye
                eye_region = frame[max(0, eye_center_y-15):eye_center_y+15,
                                 max(0, eye_center_x-20):eye_center_x+20]
                
                if eye_region.size > 0:
                    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
                    avg_brightness = np.mean(gray_eye)
                    return avg_brightness < 25 
                return False
            
            if check_eye_obstruction(left_eye, "left") and check_eye_obstruction(right_eye, "right"):
                issues.append("Eyes obstructed")
            
            # should be looking at camera
            try:
                gaze_issues = self.check_eye_gaze_direction(landmarks, frame)
                if gaze_issues:
                    issues.extend(gaze_issues)
            except:
                pass
                    
        except Exception:
            pass
            
        return issues
    

    #overall compliance
    def get_compliance_status(self, landmarks, frame):
        compliance = {
            'eyeglasses_detected': False,
            'face_coverage_issues': [],
            'compliant': True,
            'issues': []
        }
        
        try:
            coverage_issues = self.check_face_coverage(landmarks, frame)
            compliance['face_coverage_issues'] = coverage_issues
            
            compliance['eyeglasses_detected'] = any("Eyeglasses" in issue for issue in coverage_issues)
            
            compliance['issues'].extend(coverage_issues)
            
        except Exception as e:
            compliance['face_coverage_issues'] = []
            compliance['eyeglasses_detected'] = False
        
        compliance['compliant'] = len(compliance['issues']) == 0
        
        return compliance
    
    # Quality check
    def analyze_captured_photo_quality(self, photo_frame, landmarks):
        quality_issues = []
        
        print("\n" + "="*60)
        print("PHOTO QUALITY ANALYSIS")
        print("="*60)
        
        try:
            # check photo for obstructions and quality
            h, w = photo_frame.shape[:2]
            print(f"Image dimensions: {w}x{h}")
            
            # For face obstructions (glasses, bangs, masks, etc.) #
            print("\n1. Face Occlusion Result:")
            try:
                coverage_issues = self.check_face_coverage(landmarks, photo_frame)
                print(f"   Total obstructions found: {len(coverage_issues)}")
                
                if coverage_issues:
                    for i, issue in enumerate(coverage_issues, 1):
                        print(f" - {i}. {issue}")
                    quality_issues.extend(coverage_issues)
                else:
                    print("No face obstructions detected")
                    print("No eyeglasses detected")
                    print("No obstruction covering forehead")
                    print("No masks or objects detected in the face")
                    print("Eyes visible")
                    
            except Exception as e:
                print(f"Error in face obstruction analysis: {e}")
            
            # Check photo sharpness/blur #
            print("\n2. Sharpness/Blur Result:")
            try:
                gray = cv2.cvtColor(photo_frame, cv2.COLOR_BGR2GRAY)
                # calculate Laplacian variance (sharpness metric)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                print(f"Laplacian variance: {laplacian_var:.2f}")
                print(f"Threshold: 50 (higher = sharper)")
                
                # If photo is very blurry
                if laplacian_var < 50:  
                    quality_issues.append("Photo appears blurry or out of focus")
                    print(f"Error: Photo is blurry (variance: {laplacian_var:.2f} < 50)")
                else:
                    print(f"Photo is sharp (variance: {laplacian_var:.2f} >= 50)")
            except Exception as e:
                print(f"Error in sharpness analysis: {e}")
            
            print("\n3. Brightness/Contrast Result:")
            try:
                gray = cv2.cvtColor(photo_frame, cv2.COLOR_BGR2GRAY)
                
                # check photo brightness
                mean_brightness = np.mean(gray)
                print(f"   Mean brightness: {mean_brightness:.2f}")
                print(f"   Acceptable range: 50-200")
                
                if mean_brightness < 50:
                    quality_issues.append("Photo is too dark")
                    print(f"Photo too dark (brightness: {mean_brightness:.2f} < 50)")
                elif mean_brightness > 200:
                    quality_issues.append("Photo is overexposed")
                    print(f"Photo overexposed (brightness: {mean_brightness:.2f} > 200)")
                else:
                    print(f"Brightness acceptable ({mean_brightness:.2f})")
                    
                # Check contrast
                contrast = np.std(gray)
                print(f"Contrast (std dev): {contrast:.2f}")
                print(f"Minimum threshold: 30")
                
                if contrast < 30:
                    quality_issues.append("Photo has low contrast")
                    print(f"Low contrast (std: {contrast:.2f} < 30)")
                else:
                    print(f"Contrast acceptable (std: {contrast:.2f} >= 30)")
            except Exception as e:
                print(f"Error in brightness/contrast analysis: {e}")
                
        except Exception as e:
            print(f"\nError in photo quality analysis: {e}")
            
        # Summary
        print("\n" + "="*60)
        print("Quality Check Summary")
        print("="*60)
        print(f"Total issues found: {len(quality_issues)}")
        
        if quality_issues:
            print("Quality Issues:")
            for i, issue in enumerate(quality_issues, 1):
                print(f"   {i}. {issue}")
            print("\nRecommendation: Recapture photo")
        else:
            print("Photo Quality: Approved")
            print("Recommendation: Accept Photo")
        
        print("="*60)
        
        return quality_issues