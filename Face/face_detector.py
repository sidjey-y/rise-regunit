import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import os
import time

class FaceDetector:
    def __init__(self):
        # dlib's face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        
        # Check if landmarks file exists
        landmarks_file = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(landmarks_file):
            raise FileNotFoundError(f"Landmarks file not found: {landmarks_file}")
        
        self.predictor = dlib.shape_predictor(landmarks_file) #hugging face
        
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
        
        # Track initialization status
        self._initialized = True
        
        # head position estimation points, for pnp
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip, ref point
            (0.0, -330.0, -65.0),        # Chin, straight down slightly back
            (-225.0, 170.0, -135.0),     # Left eye left corner, > left, up, back (close open)
            (225.0, 170.0, -135.0),      # Right eye right corner > right up, back 
            (-150.0, -150.0, -125.0),    # Left Mouth corner > left, down, back
            (150.0, -150.0, -125.0)      # Right mouth corner > right, down, back
        ])
    
    def initialize(self) -> bool:
        """Initialize the face detector (already done in __init__)"""
        return self._initialized
    
    def is_initialized(self) -> bool:
        """Check if the face detector is properly initialized"""
        return self._initialized
    
    def cleanup(self):
        """Cleanup resources"""
        # dlib objects don't need explicit cleanup
        self._initialized = False
    
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
        """Calculate Eye Aspect Ratio with improved error handling"""
        try:
            # Validate input
            if eye is None or len(eye) < 6:
                print(f"Invalid eye landmarks: {eye}")
                return None
            
            # Check for invalid coordinates
            for point in eye:
                if len(point) != 2 or np.isnan(point[0]) or np.isnan(point[1]) or np.isinf(point[0]) or np.isinf(point[1]):
                    print(f"Invalid eye point: {point}")
                    return None
            
            #using euclidean distances between the:

            # two sets of vertical eye, x and y coordinates, vertical
            A = dist.euclidean(eye[1], eye[5])
            B = dist.euclidean(eye[2], eye[4])
            
            # between the horizontal eye landmark (x, y)-coordinates
            C = dist.euclidean(eye[0], eye[3])
            
            # Validate distances
            if A <= 0 or B <= 0 or C <= 0:
                print(f"Invalid distances: A={A}, B={B}, C={C}")
                return None
            
            # eye aspect ratio, larger> eye open > higher ear value
            ear = (A + B) / (2.0 * C)
            
            # Validate result
            if np.isnan(ear) or np.isinf(ear) or ear <= 0:
                print(f"Invalid EAR result: {ear}")
                return None
            
            return ear
            
        except Exception as e:
            print(f"Error calculating EAR: {e}")
            return None
    
    #BASED ON EAR
    def is_blinking(self, landmarks):
        """Detect blinking with improved error handling"""
        try:
            # Validate landmarks
            if landmarks is None or len(landmarks) < 48:
                print(f"Invalid landmarks for blink detection: {landmarks is None}, length: {len(landmarks) if landmarks is not None else 0}")
                return False, None
            
            # extract eye coordinates
            left_eye = landmarks[self.LEFT_EYE_START:self.LEFT_EYE_END]
            right_eye = landmarks[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
            
            #calculate EAR for both eyes
            left_ear = self.calculate_ear(left_eye)
            right_ear = self.calculate_ear(right_eye)
            
            # Check if EAR calculation failed
            if left_ear is None or right_ear is None:
                print("EAR calculation failed for one or both eyes")
                return False, None
            
            #average
            ear = (left_ear + right_ear) / 2.0
            
            # Validate final EAR
            if np.isnan(ear) or np.isinf(ear) or ear <= 0:
                print(f"Invalid final EAR: {ear}")
                return False, None
            
            #threshold for ear - more lenient threshold
            return ear < 0.25, ear  # Increased from 0.22 to 0.25 for better sensitivity
            
        except Exception as e:
            print(f"Error in is_blinking: {e}")
            return False, None
    
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
    
    def draw_landmarks(self, frame, landmarks, scale_factor=1.0):
        """Draw facial landmarks with optional scaling"""
        try:
            for (x, y) in landmarks:
                # Scale coordinates if needed
                scaled_x = int(x * scale_factor)
                scaled_y = int(y * scale_factor)
                cv2.circle(frame, (scaled_x, scaled_y), max(1, int(1 * scale_factor)), (0, 255, 0), -1)
            return frame
        except Exception as e:
            print(f"Error drawing landmarks: {e}")
            return frame
    
    def draw_face_boundary(self, frame, face, scale_factor=1.0):
        """Draw face boundary rectangle with optional scaling"""
        try:
            #rectangle to the detected face
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            
            # Scale coordinates if needed
            scaled_x = int(x * scale_factor)
            scaled_y = int(y * scale_factor)
            scaled_w = int(w * scale_factor)
            scaled_h = int(h * scale_factor)
            
            cv2.rectangle(frame, (scaled_x, scaled_y), (scaled_x + scaled_w, scaled_y + scaled_h), (0, 255, 0), max(1, int(2 * scale_factor)))
            return frame
        except Exception as e:
            print(f"Error drawing face boundary: {e}")
            return frame
    
    def get_compliance_status(self, landmarks, frame):
        """Get compliance status for face coverage and quality"""
        compliance = {
            'eyeglasses_detected': False,
            'face_coverage_issues': [],
            'compliant': True,
            'issues': []
        }
        
        try:
            # For now, just return basic compliance
            # You can add more sophisticated checks here later
            pass
        except Exception as e:
            compliance['face_coverage_issues'] = []
            compliance['eyeglasses_detected'] = False
        
        compliance['compliant'] = len(compliance['issues']) == 0
        return compliance
    
    def analyze_captured_photo_quality(self, photo_frame, landmarks):
        """Analyze photo quality - simplified version"""
        quality_issues = []
        
        try:
            # Basic quality check - check if frame is valid
            if photo_frame is None or photo_frame.size == 0:
                quality_issues.append("Invalid photo frame")
                return quality_issues
            
            # For now, just return empty list (no issues)
            # You can add more sophisticated quality checks here later
            pass
            
        except Exception as e:
            quality_issues.append(f"Error in quality analysis: {e}")
        
        return quality_issues 