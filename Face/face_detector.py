import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
from typing import Tuple, List, Optional, Dict, Any
from base_detector import BaseDetector
from config_manager import ConfigManager
import time

class FaceDetector(BaseDetector):
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):

        super().__init__(config_manager)
        
        self.detector = None
        self.predictor = None
        self.landmarks_file = None
        
        self.eye_indices = {}
        
        self.model_points = None
        
        self._last_detection = None
        self._last_landmarks = None
        
        # Performance optimization: cache head pose calculations
        self._last_head_pose = None
        self._last_head_pose_time = 0
        self._head_pose_cache_duration = 0.05  # Cache for 50ms (20 FPS for head pose)
    
    def _initialize_components(self) -> None:
        # Simple, direct initialization like face_detector_try.py
        try:
            # Get config values with fallbacks (like reference file)
            if self.config_manager:
                face_config = self.config_manager.get_face_detection_config()
                eye_config = self.config_manager.get_eye_detection_config()
                self.landmarks_file = face_config.get('landmarks_file', 'shape_predictor_68_face_landmarks.dat')
            else:
                self.landmarks_file = 'shape_predictor_68_face_landmarks.dat'
            
            # Direct initialization like your reference file
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(self.landmarks_file)
            
            # Eye indices from config or defaults (like reference file)
            self.LEFT_EYE_START = 36
            self.LEFT_EYE_END = 42
            self.RIGHT_EYE_START = 42
            self.RIGHT_EYE_END = 48
            self.NOSE_TIP = 30
            
            if self.config_manager:
                eye_config = self.config_manager.get_eye_detection_config()
                self.LEFT_EYE_START = eye_config.get('left_eye_start', 36)
                self.LEFT_EYE_END = eye_config.get('left_eye_end', 42)
                self.RIGHT_EYE_START = eye_config.get('right_eye_start', 42)
                self.RIGHT_EYE_END = eye_config.get('right_eye_end', 48)
            
            # Head pose model points (same as reference file)
            self.model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -330.0, -65.0),        # Chin
                (-225.0, 170.0, -135.0),     # Left eye left corner
                (225.0, 170.0, -135.0),      # Right eye right corner
                (-150.0, -150.0, -125.0),    # Left Mouth corner
                (150.0, -150.0, -125.0)      # Right mouth corner
            ])
            
            self.logger.info("Face detection components initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize face detection: {e}")
            raise
    
    #detect faces and landmarks that is within the frame
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        if not self.validate_frame(frame):
            return {}
        
        processed_frame = self.preprocess_frame(frame)
        
        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.detector(gray)
        
        results = {
            'faces': faces,
            'gray': gray,
            'landmarks': [],
            'analysis': {}
        }
        
        for face in faces:
            landmarks = self.get_landmarks(gray, face)
            results['landmarks'].append(landmarks)
            
            analysis = self._analyze_face(landmarks, processed_frame)
            results['analysis'][len(results['landmarks']) - 1] = analysis
        
        self._last_detection = results
        if results['landmarks']:
            self._last_landmarks = results['landmarks'][0]
        
        return self.postprocess_results(results)
    
    def detect_faces(self, frame: np.ndarray) -> Tuple[List[dlib.rectangle], np.ndarray]:
        """Detect faces in frame and return faces and grayscale image"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        return faces, gray
    
    #extract facial landmarks for a detected face
    def get_landmarks(self, gray: np.ndarray, face: dlib.rectangle) -> np.ndarray:
        landmarks = self.predictor(gray, face)
        return face_utils.shape_to_np(landmarks)
    
    #analyze facial feature & characteristics
    def _analyze_face(self, landmarks: np.ndarray, frame: np.ndarray) -> Dict[str, Any]:
        analysis = {}
        
        analysis['blinking'] = self._analyze_blinking(landmarks)
        
        analysis['head_pose'] = self._analyze_head_pose(landmarks, frame.shape)
        
        analysis['face_coverage'] = self._analyze_face_coverage(landmarks, frame)
        
        analysis['glasses'] = self._detect_glasses(landmarks, frame)
        
        return analysis
    
    #liveness - blinking behavior (from reference file)
    def _analyze_blinking(self, landmarks: np.ndarray) -> Dict[str, Any]:
        # Simple approach like reference file
        left_eye = landmarks[self.LEFT_EYE_START:self.LEFT_EYE_END]
        right_eye = landmarks[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
        
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Get threshold from config or use default like reference file
        ear_threshold = 0.25
        if self.config_manager:
            ear_threshold = self.config_manager.get('eye_detection', {}).get('ear_threshold', 0.25)
        
        return {
            'is_blinking': avg_ear < ear_threshold,
            'left_ear': left_ear,
            'right_ear': right_ear,
            'avg_ear': avg_ear,
            'threshold': ear_threshold
        }
    
    #calculate ear
    def _calculate_ear(self, eye: np.ndarray) -> float:
        """Calculate Eye Aspect Ratio with proper error handling"""
        try:
            # Validate eye array has enough points
            if eye is None or len(eye) < 6:
                return 0.0
            
            # Check for invalid coordinates (negative or NaN values)
            for point in eye[:6]:
                if (point is None or 
                    np.any(np.isnan(point)) or 
                    np.any(np.isinf(point)) or
                    point[0] < 0 or point[1] < 0):
                    return 0.0
            
            #vertical distances
            A = dist.euclidean(eye[1], eye[5])
            B = dist.euclidean(eye[2], eye[4])
            
            #horizontal distance
            C = dist.euclidean(eye[0], eye[3])
            
            # Prevent division by zero
            if C <= 0:
                return 0.0
            
            ear = (A + B) / (2.0 * C)
            
            # Validate result
            if np.isnan(ear) or np.isinf(ear):
                return 0.0
                
            return ear
            
        except Exception as e:
            print(f"Error calculating EAR: {e}")
            return 0.0
    
    #analyze head pose and orientation
    def _analyze_head_pose(self, landmarks: np.ndarray, frame_shape: Tuple[int, ...]) -> Dict[str, Any]:
        h, w = frame_shape[:2]
        
        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        
        dist_coeffs = np.zeros((4, 1))
        
        head_config = self.config_manager.get_head_pose_config()
        image_points = np.array([
            landmarks[head_config.get('nose_tip_index', 30)],     #nose tip
            landmarks[head_config.get('chin_index', 8)],          #chin
            landmarks[head_config.get('left_eye_corner', 36)],    #left eye left corner
            landmarks[head_config.get('right_eye_corner', 45)],   #right eye right corner
            landmarks[head_config.get('left_mouth_corner', 48)],  #left mouth corner
            landmarks[head_config.get('right_mouth_corner', 54)]  #right mouth corner
        ], dtype="double")
        
        #solve pnp
        success, rotation_vec, translation_vec = cv2.solvePnP(
            self.model_points, image_points, camera_matrix, dist_coeffs
        )
        
        if success:
            rotation_mat, _ = cv2.Rodrigues(rotation_vec)
            
            #get euler angles
            pose_mat = cv2.hconcat((rotation_mat, translation_vec))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            
            return {
                'success': True,
                'rotation': rotation_vec.flatten(),
                'translation': translation_vec.flatten(),
                'euler_angles': euler_angles.flatten(),
                'pitch': euler_angles[0],
                'yaw': euler_angles[1],
                'roll': euler_angles[2]
            }
        else:
            return {'success': False}
    
    #face coverage and visibility
    def _analyze_face_coverage(self, landmarks: np.ndarray, frame: np.ndarray) -> Dict[str, Any]:

        coverage_config = self.config_manager.get('face_coverage', {})
        
        face_rect = cv2.boundingRect(landmarks)
        face_area = face_rect[2] * face_rect[3]
        frame_area = frame.shape[0] * frame.shape[1]
        face_ratio = face_area / frame_area
        
        return {
            'face_ratio': face_ratio,
            'is_adequately_visible': face_ratio >= coverage_config.get('min_face_visibility', 0.8),
            'face_rect': face_rect
        }
    
    #glasses detection
    def _detect_glasses(self, landmarks: np.ndarray, frame: np.ndarray) -> Dict[str, Any]:
        """Detect glasses with consistent eye index usage"""
        try:
            coverage_config = self.config_manager.get('face_coverage', {})
            
            # Use the same eye indices as other methods
            left_eye = landmarks[self.LEFT_EYE_START:self.LEFT_EYE_END]
            right_eye = landmarks[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
            
            eye_coverage = 0.0  # placeholder for actual calculation
            
            return {
                'has_glasses': eye_coverage > coverage_config.get('max_glasses_coverage', 0.3),
                'coverage_ratio': eye_coverage
            }
        except Exception as e:
            print(f"Error in _detect_glasses: {e}")
            return {
                'has_glasses': False,
                'coverage_ratio': 0.0
            }
    
    #draw landmarks
    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:

        display_config = self.config_manager.get_display_config()
        
        if not display_config.get('show_landmarks', True):
            return frame
        
        result_frame = frame.copy()
        
        for (x, y) in landmarks:
            cv2.circle(result_frame, (x, y), 1, (0, 255, 0), -1)
        
        return result_frame
    

    #face boundary rectangle
    def draw_face_boundary(self, frame: np.ndarray, face: dlib.rectangle) -> np.ndarray:

        display_config = self.config_manager.get_display_config()
        
        if not display_config.get('show_face_boundary', True):
            return frame
        
        result_frame = frame.copy()
        
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return result_frame
    

    def get_last_detection(self) -> Optional[Dict[str, Any]]:
        return self._last_detection
    
    def get_last_landmarks(self) -> Optional[np.ndarray]:
        return self._last_landmarks
    
    def get_compliance_status(self, landmarks: np.ndarray, frame: np.ndarray) -> Dict[str, Any]:
        """Get overall compliance status for face detection (using reference file approach)"""
        compliance = {
            'eyeglasses_detected': False,
            'face_coverage_issues': [],
            'compliant': True,
            'issues': []
        }
        
        try:
            # Check if compliance detection is enabled (can be disabled for testing)
            if hasattr(self, 'config_manager') and self.config_manager:
                compliance_enabled = self.config_manager.get('compliance', {}).get('enabled', True)
                if not compliance_enabled:
                    return compliance
            
            # Use the working coverage check from reference file
            coverage_issues = self.check_face_coverage(landmarks, frame)
            compliance['face_coverage_issues'] = coverage_issues
            
            # Check for eyeglasses specifically
            compliance['eyeglasses_detected'] = any("Eyeglasses" in issue for issue in coverage_issues)
            
            # Add all coverage issues to the main issues list
            compliance['issues'].extend(coverage_issues)
            
        except Exception as e:
            # If analysis fails, assume non-compliant for safety
            compliance['face_coverage_issues'] = ["Analysis failed"]
            compliance['issues'] = ["Analysis failed"]
        
        compliance['compliant'] = len(compliance['issues']) == 0
        return compliance
    
    def check_face_coverage(self, landmarks: np.ndarray, frame: np.ndarray) -> List[str]:
        """Face coverage check from reference file"""
        issues = []
        
        try:
            # Glasses detection using reference file method
            try:
                glasses_detected = self.glasses_detection(landmarks, frame)
                if glasses_detected:
                    issues.append("Eyeglasses detected")
            except:
                pass
            
            # Get configuration values for forehead detection
            config = getattr(self, 'config_manager', None)
            if config:
                forehead_config = config.get('compliance', {}).get('forehead', {})
                region_height = forehead_config.get('region_height', 40)
                hair_threshold = forehead_config.get('hair_threshold', 0.80)
            else:
                region_height = 40
                hair_threshold = 0.80
            
            # Forehead obstruction check (from reference file)
            eyebrow_points = landmarks[17:27]  # eyebrow landmarks
            if len(eyebrow_points) > 0:
                forehead_top_y = int(min(eyebrow_points[:, 1])) - region_height
                forehead_bottom_y = int(min(eyebrow_points[:, 1]))
                face_left = int(min(landmarks[:, 0]))
                face_right = int(max(landmarks[:, 0]))
                
                if forehead_top_y >= 0 and forehead_bottom_y > forehead_top_y:
                    forehead_region = frame[forehead_top_y:forehead_bottom_y, face_left:face_right]
                    
                    if forehead_region.size > 0:
                        gray_forehead = cv2.cvtColor(forehead_region, cv2.COLOR_BGR2GRAY)
                        mean_brightness = np.mean(gray_forehead)
                        adaptive_threshold = max(50, min(80, int(mean_brightness * 0.7)))
                        _, thresh = cv2.threshold(gray_forehead, adaptive_threshold, 255, cv2.THRESH_BINARY)
                        
                        hair_pixels = np.sum(thresh == 0)
                        total_pixels = thresh.size
                        hair_ratio = hair_pixels / total_pixels if total_pixels > 0 else 0
                        
                        # More conservative threshold - only detect if there's significant hair coverage
                        # This reduces false positives from natural shadows and slight hair
                        if hair_ratio > hair_threshold:
                            issues.append("Obstruction covering forehead")
                            
        except Exception:
            pass
            
        return issues
    
    def debug_compliance_detection(self, landmarks: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Debug method to visualize compliance detection regions"""
        debug_frame = frame.copy()
        
        try:
            # Draw glasses detection region
            left_eye = landmarks[self.LEFT_EYE_START:self.LEFT_EYE_END]
            right_eye = landmarks[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
            
            if len(left_eye) > 0 and len(right_eye) > 0:
                left_eye_y = int(np.mean(left_eye[:, 1]))
                right_eye_y = int(np.mean(right_eye[:, 1]))
                glasses_y = (left_eye_y + right_eye_y) // 2
                
                left_x = int(min(left_eye[:, 0])) - 10
                right_x = int(max(right_eye[:, 0])) + 10
                strip_top = max(0, glasses_y - 4)
                strip_bottom = min(frame.shape[0], glasses_y + 4)
                
                # Draw glasses detection region
                cv2.rectangle(debug_frame, (left_x, strip_top), (right_x, strip_bottom), (0, 255, 255), 2)
                cv2.putText(debug_frame, "Glasses Detection", (left_x, strip_top - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Draw forehead detection region
            eyebrow_points = landmarks[17:27]
            if len(eyebrow_points) > 0:
                forehead_top_y = int(min(eyebrow_points[:, 1])) - 40
                forehead_bottom_y = int(min(eyebrow_points[:, 1]))
                face_left = int(min(landmarks[:, 0]))
                face_right = int(max(landmarks[:, 0]))
                
                if forehead_top_y >= 0 and forehead_bottom_y > forehead_top_y:
                    # Draw forehead detection region
                    cv2.rectangle(debug_frame, (face_left, forehead_top_y), (face_right, forehead_bottom_y), (255, 0, 255), 2)
                    cv2.putText(debug_frame, "Forehead Detection", (face_left, forehead_top_y - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    
        except Exception as e:
            print(f"Error in debug visualization: {e}")
        
        return debug_frame
    
    def glasses_detection(self, landmarks: np.ndarray, frame: np.ndarray) -> bool:
        """Glasses detection from reference file"""
        try:
            left_eye = landmarks[self.LEFT_EYE_START:self.LEFT_EYE_END]
            right_eye = landmarks[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
            
            # More precise region around eyes where glasses frames would be
            left_eye_y = int(np.mean(left_eye[:, 1]))
            right_eye_y = int(np.mean(right_eye[:, 1]))
            glasses_y = (left_eye_y + right_eye_y) // 2
            
            # Get configuration values for glasses detection
            config = getattr(self, 'config_manager', None)
            if config:
                glasses_config = config.get('compliance', {}).get('glasses', {})
                margin = glasses_config.get('region_margin', 10)
                strip_height = glasses_config.get('strip_height', 4)
            else:
                margin = 10
                strip_height = 4
            
            # X coordinates spanning both eyes with configurable margin
            left_x = int(min(left_eye[:, 0])) - margin
            right_x = int(max(right_eye[:, 0])) + margin
            
            # Thinner strip to focus on actual glasses frame area
            strip_top = max(0, glasses_y - strip_height)
            strip_bottom = min(frame.shape[0], glasses_y + strip_height)
            strip_left = max(0, left_x)
            strip_right = min(frame.shape[1], right_x)
            
            glasses_strip = frame[strip_top:strip_bottom, strip_left:strip_right]
            
            if glasses_strip.size == 0:
                return False
            
            # Convert to grayscale
            gray_strip = cv2.cvtColor(glasses_strip, cv2.COLOR_BGR2GRAY)
            
            # Use adaptive threshold for better edge detection
            thresh = cv2.adaptiveThreshold(gray_strip, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            
            # Look for horizontal lines that could be glasses frames
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
            horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Count horizontal line pixels
            horizontal_pixels = np.sum(horizontal_lines > 0)
            total_pixels = horizontal_lines.size
            
            # Calculate horizontal line ratio
            horizontal_ratio = horizontal_pixels / total_pixels if total_pixels > 0 else 0
            
            # Get threshold from configuration
            threshold = 0.15  # Default
            if config:
                threshold = glasses_config.get('threshold', 0.15)
            
            # Much more conservative threshold - only detect if there's a strong horizontal line
            # This reduces false positives from natural shadows and facial features
            return horizontal_ratio > threshold
            
        except Exception:
            return False
    
    def is_blinking(self, landmarks: np.ndarray) -> Tuple[bool, float]:
        """Check if person is blinking with proper error handling"""
        try:
            # Validate landmarks
            if landmarks is None or len(landmarks) < 48:  # Need at least 48 landmarks for eyes
                return False, 0.0
            
            # Check for invalid landmark coordinates
            if np.any(np.isnan(landmarks)) or np.any(np.isinf(landmarks)):
                return False, 0.0
            
            # Extract eye coordinates with bounds checking
            left_eye_start = min(self.LEFT_EYE_START, len(landmarks) - 1)
            left_eye_end = min(self.LEFT_EYE_END, len(landmarks))
            right_eye_start = min(self.RIGHT_EYE_START, len(landmarks) - 1)
            right_eye_end = min(self.RIGHT_EYE_END, len(landmarks))
            
            left_eye = landmarks[left_eye_start:left_eye_end]
            right_eye = landmarks[right_eye_start:right_eye_end]
            
            # Validate eye arrays
            if len(left_eye) < 6 or len(right_eye) < 6:
                return False, 0.0
            
            # Calculate EAR for both eyes
            left_ear = self._calculate_ear(left_eye)
            right_ear = self._calculate_ear(right_eye)
            
            # Check if EAR calculation failed
            if left_ear <= 0 or right_ear <= 0:
                return False, 0.0
            
            # Average EAR
            ear = (left_ear + right_ear) / 2.0
            
            # Validate final EAR value
            if np.isnan(ear) or np.isinf(ear) or ear < 0:
                return False, 0.0
            
            # Threshold for blinking (from reference file)
            return ear < 0.25, ear
            
        except Exception as e:
            print(f"Error in is_blinking: {e}")
            return False, 0.0
    
    def get_face_direction(self, landmarks: np.ndarray) -> str:
        """Get face direction (from reference file)"""
        # Nose tip and face boundaries
        nose_tip = landmarks[self.NOSE_TIP]
        
        # Face bounding box from landmarks
        left_face = min(landmarks[:, 0])
        right_face = max(landmarks[:, 0])
        face_width = right_face - left_face
        face_center_x = left_face + face_width / 2
        
        # Relative position of nose tip
        nose_relative = (nose_tip[0] - face_center_x) / (face_width / 2)
        
        # Threshold for direction detection (from reference file)
        if nose_relative < -0.15:
            return "left"
        elif nose_relative > 0.15:
            return "right"
        else:
            return "center"
    
    def get_head_pose(self, landmarks: np.ndarray, frame_shape: Tuple[int, int, int]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Get head pose with improved accuracy, error handling, and performance caching"""
        current_time = time.time()
        
        # Check if we can use cached result
        if (self._last_head_pose is not None and 
            current_time - self._last_head_pose_time < self._head_pose_cache_duration):
            return self._last_head_pose
        
        try:
            h, w = frame_shape[:2]
            
            # Improved focal length calculation
            focal_length = max(w, h)  # Use the larger dimension for better accuracy
            center = (w/2, h/2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype="double")
            
            dist_coeffs = np.zeros((4, 1))
            
            # Ensure landmarks are valid
            if (landmarks[30][0] < 0 or landmarks[30][1] < 0 or 
                landmarks[8][0] < 0 or landmarks[8][1] < 0 or
                landmarks[36][0] < 0 or landmarks[36][1] < 0 or
                landmarks[45][0] < 0 or landmarks[45][1] < 0 or
                landmarks[48][0] < 0 or landmarks[48][1] < 0 or
                landmarks[54][0] < 0 or landmarks[54][1] < 0):
                return None, None, None
            
            image_points = np.array([
                landmarks[30],     # Nose tip
                landmarks[8],      # Chin
                landmarks[36],     # Left eye left corner
                landmarks[45],     # Right eye right corner
                landmarks[48],     # Left mouth corner
                landmarks[54]      # Right mouth corner
            ], dtype="double")
            
            # Solve PnP with better error handling
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # Rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                
                # Extract Euler angles with improved calculation
                sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
                singular = sy < 1e-6
                
                if not singular:
                    x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
                    y = np.arctan2(-rotation_matrix[2,0], sy)
                    z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
                else:
                    x = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
                    y = np.arctan2(-rotation_matrix[2,0], sy)
                    z = 0
                
                # Convert to degrees
                pitch = np.degrees(x)  # up/down
                yaw = np.degrees(y)    # left/right (negative = left, positive = right)
                roll = np.degrees(z)   # tilt
                
                # Clamp values to reasonable ranges
                pitch = np.clip(pitch, -90, 90)
                yaw = np.clip(yaw, -90, 90)
                roll = np.clip(roll, -90, 90)
                
                # Cache the result
                result = (pitch, yaw, roll)
                self._last_head_pose = result
                self._last_head_pose_time = current_time
                
                return result
            
        except Exception as e:
            print(f"Head pose calculation error: {e}")
            return None, None, None
        
        return None, None, None
    
    def analyze_captured_photo_quality(self, photo_frame: np.ndarray, landmarks: np.ndarray) -> List[str]:
        """Analyze the quality of a captured photo (using reference file approach)"""
        quality_issues = []
        
        print("\n" + "="*60)
        print("PHOTO QUALITY ANALYSIS")
        print("="*60)
        
        try:
            # Check photo for obstructions and quality
            h, w = photo_frame.shape[:2]
            print(f"Image dimensions: {w}x{h}")
            
            # Face obstructions check using reference file method
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
            
            # Check photo sharpness/blur
            print("\n2. Sharpness/Blur Result:")
            try:
                gray = cv2.cvtColor(photo_frame, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                print(f"Laplacian variance: {laplacian_var:.2f}")
                print(f"Threshold: 50 (higher = sharper)")
                
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
    
    def cleanup(self) -> None:
        self.detector = None
        self.predictor = None
        self._last_detection = None
        self._last_landmarks = None
        super().cleanup() 

    def check_face_obstruction_compliance(self, landmarks: np.ndarray, frame: np.ndarray) -> Dict[str, Any]:
        """Comprehensive face obstruction compliance checking based on YAML configuration"""
        compliance_result = {
            'compliant': True,
            'issues': [],
            'details': {
                'glasses': {'compliant': True, 'issues': []},
                'hair': {'compliant': True, 'issues': []},
                'face_coverings': {'compliant': True, 'issues': []},
                'accessories': {'compliant': True, 'issues': []},
                'overall_visibility': {'compliant': True, 'issues': []}
            }
        }
        
        try:
            # Get compliance configuration
            if not hasattr(self, 'config_manager') or not self.config_manager:
                return compliance_result
                
            compliance_config = self.config_manager.get('compliance', {})
            face_obstruction_config = compliance_config.get('face_obstruction', {})
            
            if not face_obstruction_config.get('enabled', False):
                return compliance_result
            
            # 1. Glasses Compliance Check
            if face_obstruction_config.get('glasses_compliance', {}).get('allowed', True):
                glasses_compliance = self._check_glasses_compliance(landmarks, frame, face_obstruction_config)
                compliance_result['details']['glasses'] = glasses_compliance
                if not glasses_compliance['compliant']:
                    compliance_result['compliant'] = False
                    compliance_result['issues'].extend(glasses_compliance['issues'])
            
            # 2. Hair Obstruction Compliance Check
            if face_obstruction_config.get('hair_compliance', {}).get('allowed', True):
                hair_compliance = self._check_hair_compliance(landmarks, frame, face_obstruction_config)
                compliance_result['details']['hair'] = hair_compliance
                if not hair_compliance['compliant']:
                    compliance_result['compliant'] = False
                    compliance_result['issues'].extend(hair_compliance['issues'])
            
            # 3. Face Coverings Compliance Check
            face_coverings_compliance = self._check_face_coverings_compliance(landmarks, frame, face_obstruction_config)
            compliance_result['details']['face_coverings'] = face_coverings_compliance
            if not face_coverings_compliance['compliant']:
                compliance_result['compliant'] = False
                compliance_result['issues'].extend(face_coverings_compliance['issues'])
            
            # 4. Accessories Compliance Check
            accessories_compliance = self._check_accessories_compliance(landmarks, frame, face_obstruction_config)
            compliance_result['details']['accessories'] = accessories_compliance
            if not accessories_compliance['compliant']:
                compliance_result['compliant'] = False
                compliance_result['issues'].extend(accessories_compliance['issues'])
            
            # 5. Overall Visibility Compliance Check
            visibility_compliance = self._check_overall_visibility_compliance(landmarks, frame, face_obstruction_config)
            compliance_result['details']['overall_visibility'] = visibility_compliance
            if not visibility_compliance['compliant']:
                compliance_result['compliant'] = False
                compliance_result['issues'].extend(visibility_compliance['issues'])
                
        except Exception as e:
            compliance_result['compliant'] = False
            compliance_result['issues'].append(f"Compliance check failed: {str(e)}")
            
        return compliance_result
    
    def _check_glasses_compliance(self, landmarks: np.ndarray, frame: np.ndarray, config: Dict) -> Dict[str, Any]:
        """Check glasses compliance based on configuration"""
        result = {'compliant': True, 'issues': []}
        
        try:
            glasses_config = config.get('glasses_compliance', {})
            max_coverage = glasses_config.get('max_coverage', 0.30)
            max_thickness = glasses_config.get('frame_thickness', 0.05)
            
            # Check if glasses are detected
            glasses_detected = self.glasses_detection(landmarks, frame)
            
            if glasses_detected:
                # Calculate glasses coverage
                coverage = self._calculate_glasses_coverage(landmarks, frame)
                thickness = self._calculate_frame_thickness(landmarks, frame)
                
                if coverage > max_coverage:
                    result['compliant'] = False
                    result['issues'].append(f"Glasses coverage ({coverage:.2f}) exceeds limit ({max_coverage})")
                
                if thickness > max_thickness:
                    result['compliant'] = False
                    result['issues'].append(f"Frame thickness ({thickness:.2f}) exceeds limit ({max_thickness})")
                    
        except Exception as e:
            result['compliant'] = False
            result['issues'].append(f"Glasses compliance check failed: {str(e)}")
            
        return result
    
    def _check_hair_compliance(self, landmarks: np.ndarray, frame: np.ndarray, config: Dict) -> Dict[str, Any]:
        """Check hair obstruction compliance based on configuration"""
        result = {'compliant': True, 'issues': []}
        
        try:
            hair_config = config.get('hair_compliance', {})
            max_forehead_coverage = hair_config.get('max_forehead_coverage', 0.25)
            max_eye_coverage = hair_config.get('max_eye_coverage', 0.10)
            
            # Check forehead hair coverage
            forehead_coverage = self._calculate_forehead_hair_coverage(landmarks, frame)
            if forehead_coverage > max_forehead_coverage:
                result['compliant'] = False
                result['issues'].append(f"Forehead hair coverage ({forehead_coverage:.2f}) exceeds limit ({max_forehead_coverage})")
            
            # Check eye hair coverage
            eye_coverage = self._calculate_eye_hair_coverage(landmarks, frame)
            if eye_coverage > max_eye_coverage:
                result['compliant'] = False
                result['issues'].append(f"Eye hair coverage ({eye_coverage:.2f}) exceeds limit ({max_eye_coverage})")
                
        except Exception as e:
            result['compliant'] = False
            result['issues'].append(f"Hair compliance check failed: {str(e)}")
            
        return result
    
    def _check_face_coverings_compliance(self, landmarks: np.ndarray, frame: np.ndarray, config: Dict) -> Dict[str, Any]:
        """Check face coverings compliance based on configuration"""
        result = {'compliant': True, 'issues': []}
        
        try:
            coverings_config = config.get('face_coverings', {})
            masks_allowed = coverings_config.get('masks_allowed', False)
            scarves_allowed = coverings_config.get('scarves_allowed', False)
            max_face_coverage = coverings_config.get('max_face_coverage', 0.15)
            
            # Check for masks
            mask_detected = self._detect_face_mask(landmarks, frame)
            if mask_detected and not masks_allowed:
                result['compliant'] = False
                result['issues'].append("Face masks not allowed")
            
            # Check for scarves
            scarf_detected = self._detect_scarf(landmarks, frame)
            if scarf_detected and not scarves_allowed:
                result['compliant'] = False
                result['issues'].append("Scarves not allowed")
            
            # Check overall face coverage
            face_coverage = self._calculate_face_coverage(landmarks, frame)
            if face_coverage > max_face_coverage:
                result['compliant'] = False
                result['issues'].append(f"Face coverage ({face_coverage:.2f}) exceeds limit ({max_face_coverage})")
                
        except Exception as e:
            result['compliant'] = False
            result['issues'].append(f"Face coverings compliance check failed: {str(e)}")
            
        return result
    
    def _check_accessories_compliance(self, landmarks: np.ndarray, frame: np.ndarray, config: Dict) -> Dict[str, Any]:
        """Check accessories compliance based on configuration"""
        result = {'compliant': True, 'issues': []}
        
        try:
            accessories_config = config.get('accessories', {})
            hats_allowed = accessories_config.get('hats_allowed', False)
            sunglasses_allowed = accessories_config.get('sunglasses_allowed', False)
            max_jewelry_coverage = accessories_config.get('jewelry_coverage', 0.05)
            
            # Check for hats
            hat_detected = self._detect_hat(landmarks, frame)
            if hat_detected and not hats_allowed:
                result['compliant'] = False
                result['issues'].append("Hats not allowed")
            
            # Check for sunglasses
            sunglasses_detected = self._detect_sunglasses(landmarks, frame)
            if sunglasses_detected and not sunglasses_allowed:
                result['compliant'] = False
                result['issues'].append("Sunglasses not allowed")
            
            # Check jewelry coverage
            jewelry_coverage = self._calculate_jewelry_coverage(landmarks, frame)
            if jewelry_coverage > max_jewelry_coverage:
                result['compliant'] = False
                result['issues'].append(f"Jewelry coverage ({jewelry_coverage:.2f}) exceeds limit ({max_jewelry_coverage})")
                
        except Exception as e:
            result['compliant'] = False
            result['issues'].append(f"Accessories compliance check failed: {str(e)}")
            
        return result
    
    def _check_overall_visibility_compliance(self, landmarks: np.ndarray, frame: np.ndarray, config: Dict) -> Dict[str, Any]:
        """Check overall face visibility compliance"""
        result = {'compliant': True, 'issues': []}
        
        if not config.get('enabled', True):
            return result
            
        thresholds = config.get('compliance_thresholds', {})
        
        # Calculate visibility metrics
        face_visibility = self._calculate_face_visibility(landmarks, frame)
        eye_visibility = self._calculate_eye_visibility(landmarks, frame)
        mouth_visibility = self._calculate_mouth_visibility(landmarks, frame)
        nose_visibility = self._calculate_nose_visibility(landmarks, frame)
        
        # Check against thresholds
        if face_visibility < thresholds.get('min_face_visibility', 0.85):
            result['compliant'] = False
            result['issues'].append(f"Face visibility too low: {face_visibility:.2f}")
            
        if eye_visibility < thresholds.get('min_eye_visibility', 0.90):
            result['compliant'] = False
            result['issues'].append(f"Eye visibility too low: {eye_visibility:.2f}")
            
        if mouth_visibility < thresholds.get('min_mouth_visibility', 0.80):
            result['compliant'] = False
            result['issues'].append(f"Mouth visibility too low: {mouth_visibility:.2f}")
            
        if nose_visibility < thresholds.get('min_nose_visibility', 0.85):
            result['compliant'] = False
            result['issues'].append(f"Nose visibility too low: {nose_visibility:.2f}")
            
        return result

    def _calculate_glasses_coverage(self, landmarks: np.ndarray, frame: np.ndarray) -> float:
        """Calculate the percentage of eye region covered by glasses"""
        # Eye region landmarks (27-36 for left eye, 36-45 for right eye)
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        # Create eye region with margin
        left_eye_region = self._get_eye_region(left_eye, frame, margin=15)
        right_eye_region = self._get_eye_region(right_eye, frame, margin=15)
        
        # Analyze horizontal lines in eye regions
        left_coverage = self._analyze_horizontal_lines(left_eye_region)
        right_coverage = self._analyze_horizontal_lines(right_eye_region)
        
        return (left_coverage + right_coverage) / 2.0

    def _calculate_frame_thickness(self, landmarks: np.ndarray, frame: np.ndarray) -> float:
        """Calculate the thickness of glasses frames"""
        # Focus on the bridge area between eyes
        bridge_region = self._get_bridge_region(landmarks, frame)
        
        # Analyze vertical edges to detect frame thickness
        edges = cv2.Canny(bridge_region, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
            
        # Find the largest contour (likely the frame)
        largest_contour = max(contours, key=cv2.contourArea)
        thickness = self._calculate_contour_thickness(largest_contour)
        
        # Normalize by face width
        face_width = self._get_face_width(landmarks)
        return thickness / face_width if face_width > 0 else 0.0

    def _calculate_forehead_hair_coverage(self, landmarks: np.ndarray, frame: np.ndarray) -> float:
        """Calculate hair coverage over the forehead region"""
        # Forehead region above eyebrows
        eyebrow_left = landmarks[19]  # Left eyebrow
        eyebrow_right = landmarks[24]  # Right eyebrow
        
        # Create forehead region
        forehead_region = self._get_forehead_region(landmarks, frame, height=40)
        
        # Analyze hair coverage using edge detection
        gray = cv2.cvtColor(forehead_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        
        # Calculate hair coverage ratio
        total_pixels = forehead_region.shape[0] * forehead_region.shape[1]
        hair_pixels = np.count_nonzero(edges)
        
        return hair_pixels / total_pixels if total_pixels > 0 else 0.0

    def _calculate_eye_hair_coverage(self, landmarks: np.ndarray, frame: np.ndarray) -> float:
        """Calculate hair coverage over the eyes"""
        # Eye regions with extended margins
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        
        left_eye_region = self._get_eye_region(left_eye, frame, margin=20)
        right_eye_region = self._get_eye_region(right_eye, frame, margin=20)
        
        # Analyze hair coverage in both eye regions
        left_coverage = self._analyze_hair_in_region(left_eye_region)
        right_coverage = self._analyze_hair_in_region(right_eye_region)
        
        return (left_coverage + right_coverage) / 2.0
    
    def _get_eye_region(self, eye_landmarks: np.ndarray, frame: np.ndarray, margin: int) -> np.ndarray:
        """Helper to get a region around eye landmarks with a margin."""
        left_x = int(min(eye_landmarks[:, 0])) - margin
        right_x = int(max(eye_landmarks[:, 0])) + margin
        top_y = int(min(eye_landmarks[:, 1])) - margin
        bottom_y = int(max(eye_landmarks[:, 1])) + margin
        
        # Ensure bounds
        left_x = max(0, left_x)
        right_x = min(frame.shape[1], right_x)
        top_y = max(0, top_y)
        bottom_y = min(frame.shape[0], bottom_y)
        
        return frame[top_y:bottom_y, left_x:right_x]

    def _analyze_horizontal_lines(self, image: np.ndarray) -> float:
        """Analyzes an image for horizontal lines and returns a coverage ratio."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1)) # Adjust kernel size for horizontal lines
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Count horizontal line pixels
        horizontal_pixels = np.sum(horizontal_lines > 0)
        total_pixels = horizontal_lines.size
        
        return horizontal_pixels / total_pixels if total_pixels > 0 else 0.0

    def _get_bridge_region(self, landmarks: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Helper to get a region between eye corners."""
        left_eye_corner = landmarks[36]
        right_eye_corner = landmarks[45]
        
        # Calculate center of the bridge
        bridge_center_x = (left_eye_corner[0] + right_eye_corner[0]) / 2
        bridge_center_y = (left_eye_corner[1] + right_eye_corner[1]) / 2
        
        # Define a region around the bridge
        margin = 20 # Adjust margin as needed
        left_x = int(bridge_center_x - (right_eye_corner[0] - left_eye_corner[0]) * 0.5) - margin
        right_x = int(bridge_center_x + (right_eye_corner[0] - left_eye_corner[0]) * 0.5) + margin
        top_y = int(bridge_center_y - 10) - margin # Above the bridge
        bottom_y = int(bridge_center_y + 10) + margin # Below the bridge
        
        # Ensure bounds
        left_x = max(0, left_x)
        right_x = min(frame.shape[1], right_x)
        top_y = max(0, top_y)
        bottom_y = min(frame.shape[0], bottom_y)
        
        return frame[top_y:bottom_y, left_x:right_x]

    def _calculate_contour_thickness(self, contour: np.ndarray) -> float:
        """Calculates the thickness of a contour (e.g., glasses frame)."""
        # This is a simplified approach. A more robust method would involve
        # fitting a line or curve to the contour and measuring its width.
        # For now, we'll just return a placeholder value.
        return 1.0 # Placeholder for actual thickness calculation

    def _get_face_width(self, landmarks: np.ndarray) -> float:
        """Calculates the width of the face from landmarks."""
        left_face = min(landmarks[:, 0])
        right_face = max(landmarks[:, 0])
        return right_face - left_face

    def _get_forehead_region(self, landmarks: np.ndarray, frame: np.ndarray, height: int) -> np.ndarray:
        """Helper to get a region above eyebrows."""
        eyebrow_top_y = int(min(landmarks[17:27, 1])) # Average Y of eyebrow points
        eyebrow_bottom_y = int(min(landmarks[17:27, 1])) # Average Y of eyebrow points
        
        face_left = int(min(landmarks[:, 0]))
        face_right = int(max(landmarks[:, 0]))
        
        # Ensure bounds
        eyebrow_top_y = max(0, eyebrow_top_y - height)
        eyebrow_bottom_y = max(0, eyebrow_bottom_y)
        face_left = max(0, face_left)
        face_right = min(frame.shape[1], face_right)
        
        return frame[eyebrow_top_y:eyebrow_bottom_y, face_left:face_right]

    def _analyze_hair_in_region(self, image: np.ndarray) -> float:
        """Analyzes an image for hair coverage."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Count non-black pixels (likely hair)
        hair_pixels = np.count_nonzero(thresh == 0)
        total_pixels = thresh.size
        
        return hair_pixels / total_pixels if total_pixels > 0 else 0.0
    
    def _detect_face_mask(self, landmarks: np.ndarray, frame: np.ndarray) -> bool:
        """Detect if person is wearing a face mask"""
        try:
            # Simple mask detection based on lower face coverage
            # This is a simplified approach - in production you'd use more sophisticated methods
            
            # Define mouth and nose region
            nose_points = landmarks[27:36]  # nose landmarks
            mouth_points = landmarks[48:68]  # mouth landmarks
            
            if len(nose_points) == 0 or len(mouth_points) == 0:
                return False
            
            # Calculate face dimensions
            face_width = max(landmarks[:, 0]) - min(landmarks[:, 0])
            face_height = max(landmarks[:, 1]) - min(landmarks[:, 1])
            
            # Calculate lower face region
            lower_face_top = int(min(nose_points[:, 1]))
            lower_face_bottom = int(max(mouth_points[:, 1])) + int(face_height * 0.1)
            lower_face_left = int(min(landmarks[:, 0]))
            lower_face_right = int(max(landmarks[:, 0]))
            
            # Ensure bounds
            lower_face_top = max(0, lower_face_top)
            lower_face_bottom = min(frame.shape[0], lower_face_bottom)
            lower_face_left = max(0, lower_face_left)
            lower_face_right = min(frame.shape[1], lower_face_right)
            
            if lower_face_bottom <= lower_face_top or lower_face_right <= lower_face_left:
                return False
            
            # Analyze lower face region for mask-like patterns
            lower_face_region = frame[lower_face_top:lower_face_bottom, lower_face_left:lower_face_right]
            if lower_face_region.size == 0:
                return False
            
            # Convert to grayscale and analyze
            gray_lower = cv2.cvtColor(lower_face_region, cv2.COLOR_BGR2GRAY)
            
            # Simple heuristic: if lower face is very uniform and bright, likely a mask
            std_dev = np.std(gray_lower)
            mean_brightness = np.mean(gray_lower)
            
            # Mask detection criteria (simplified)
            return std_dev < 20 and mean_brightness > 150
            
        except Exception:
            return False
    
    def _detect_scarf(self, landmarks: np.ndarray, frame: np.ndarray) -> bool:
        """Detect if person is wearing a scarf"""
        try:
            # Simplified scarf detection - looks for large coverage below mouth
            mouth_points = landmarks[48:68]  # mouth landmarks
            
            if len(mouth_points) == 0:
                return False
            
            # Define region below mouth
            below_mouth_top = int(max(mouth_points[:, 1])) + 10
            below_mouth_bottom = int(below_mouth_top + (max(landmarks[:, 1]) - min(landmarks[:, 1])) * 0.3)
            below_mouth_left = int(min(landmarks[:, 0]))
            below_mouth_right = int(max(landmarks[:, 0]))
            
            # Ensure bounds
            below_mouth_top = max(0, below_mouth_top)
            below_mouth_bottom = min(frame.shape[0], below_mouth_bottom)
            below_mouth_left = max(0, below_mouth_left)
            below_mouth_right = min(frame.shape[1], below_mouth_right)
            
            if below_mouth_bottom <= below_mouth_top or below_mouth_right <= below_mouth_left:
                return False
            
            # Analyze region below mouth
            below_mouth_region = frame[below_mouth_top:below_mouth_bottom, below_mouth_left:below_mouth_right]
            if below_mouth_region.size == 0:
                return False
            
            # Convert to grayscale and analyze
            gray_below = cv2.cvtColor(below_mouth_region, cv2.COLOR_BGR2GRAY)
            
            # Scarf detection: look for large uniform regions
            std_dev = np.std(gray_below)
            mean_brightness = np.mean(gray_below)
            
            # Scarf detection criteria (simplified)
            return std_dev < 25 and mean_brightness < 120
            
        except Exception:
            return False
    
    def _detect_hat(self, landmarks: np.ndarray, frame: np.ndarray) -> bool:
        """Detect if person is wearing a hat"""
        try:
            # Hat detection based on forehead region analysis
            eyebrow_points = landmarks[17:27]  # eyebrow landmarks
            
            if len(eyebrow_points) == 0:
                return False
            
            # Define region above eyebrows
            above_eyebrow_top = int(min(eyebrow_points[:, 1])) - 60
            above_eyebrow_bottom = int(min(eyebrow_points[:, 1])) - 20
            above_eyebrow_left = int(min(landmarks[:, 0]))
            above_eyebrow_right = int(max(landmarks[:, 0]))
            
            # Ensure bounds
            above_eyebrow_top = max(0, above_eyebrow_top)
            above_eyebrow_bottom = max(0, above_eyebrow_bottom)
            above_eyebrow_left = max(0, above_eyebrow_left)
            above_eyebrow_right = min(frame.shape[1], above_eyebrow_right)
            
            if above_eyebrow_bottom <= above_eyebrow_top or above_eyebrow_right <= above_eyebrow_left:
                return False
            
            # Analyze region above eyebrows
            above_eyebrow_region = frame[above_eyebrow_top:above_eyebrow_bottom, above_eyebrow_left:above_eyebrow_right]
            if above_eyebrow_region.size == 0:
                return False
            
            # Convert to grayscale and analyze
            gray_above = cv2.cvtColor(above_eyebrow_region, cv2.COLOR_BGR2GRAY)
            
            # Hat detection: look for large uniform regions with low brightness
            std_dev = np.std(gray_above)
            mean_brightness = np.mean(gray_above)
            
            # Hat detection criteria (simplified)
            return std_dev < 30 and mean_brightness < 100
            
        except Exception:
            return False
    
    def _detect_sunglasses(self, landmarks: np.ndarray, frame: np.ndarray) -> bool:
        """Detect if person is wearing sunglasses"""
        try:
            # Sunglasses detection based on eye region analysis
            left_eye = landmarks[self.LEFT_EYE_START:self.LEFT_EYE_END]
            right_eye = landmarks[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
            
            if len(left_eye) == 0 or len(right_eye) == 0:
                return False
            
            # Define eye regions
            left_eye_left = int(min(left_eye[:, 0])) - 5
            left_eye_right = int(max(left_eye[:, 0])) + 5
            left_eye_top = int(min(left_eye[:, 1])) - 5
            left_eye_bottom = int(max(left_eye[:, 1])) + 5
            
            right_eye_left = int(min(right_eye[:, 0])) - 5
            right_eye_right = int(max(right_eye[:, 0])) + 5
            right_eye_top = int(min(right_eye[:, 1])) - 5
            right_eye_bottom = int(max(right_eye[:, 1])) + 5
            
            # Ensure bounds
            left_eye_left = max(0, left_eye_left)
            left_eye_right = min(frame.shape[1], left_eye_right)
            left_eye_top = max(0, left_eye_top)
            left_eye_bottom = min(frame.shape[0], left_eye_bottom)
            
            right_eye_left = max(0, right_eye_left)
            right_eye_right = min(frame.shape[1], right_eye_right)
            right_eye_top = max(0, right_eye_top)
            right_eye_bottom = min(frame.shape[0], right_eye_bottom)
            
            # Analyze both eye regions
            total_dark_pixels = 0
            total_pixels = 0
            
            # Left eye
            if left_eye_bottom > left_eye_top and left_eye_right > left_eye_left:
                left_eye_region = frame[left_eye_top:left_eye_bottom, left_eye_left:left_eye_right]
                if left_eye_region.size > 0:
                    gray_left = cv2.cvtColor(left_eye_region, cv2.COLOR_BGR2GRAY)
                    # Count very dark pixels (likely sunglasses)
                    total_dark_pixels += np.sum(gray_left < 50)
                    total_pixels += gray_left.size
            
            # Right eye
            if right_eye_bottom > right_eye_top and right_eye_right > right_eye_left:
                right_eye_region = frame[right_eye_top:right_eye_bottom, right_eye_left:right_eye_right]
                if right_eye_region.size > 0:
                    gray_right = cv2.cvtColor(right_eye_region, cv2.COLOR_BGR2GRAY)
                    # Count very dark pixels (likely sunglasses)
                    total_dark_pixels += np.sum(gray_right < 50)
                    total_pixels += gray_right.size
            
            # Sunglasses detection: if significant portion is very dark
            dark_ratio = total_dark_pixels / total_pixels if total_pixels > 0 else 0.0
            return dark_ratio > 0.4  # 40% of eye region is very dark
            
        except Exception:
            return False
    
    def _calculate_jewelry_coverage(self, landmarks: np.ndarray, frame: np.ndarray) -> float:
        """Calculate jewelry coverage on face"""
        try:
            # Simplified jewelry detection - looks for small bright regions
            # This is a basic implementation - production systems would use more sophisticated methods
            
            # Define face region
            face_left = int(min(landmarks[:, 0]))
            face_right = int(max(landmarks[:, 0]))
            face_top = int(min(landmarks[:, 1]))
            face_bottom = int(max(landmarks[:, 1]))
            
            # Ensure bounds
            face_left = max(0, face_left)
            face_right = min(frame.shape[1], face_right)
            face_top = max(0, face_top)
            face_bottom = min(frame.shape[0], face_bottom)
            
            if face_bottom <= face_top or face_right <= face_left:
                return 0.0
            
            # Analyze face region for jewelry-like patterns
            face_region = frame[face_top:face_bottom, face_left:face_right]
            if face_region.size == 0:
                return 0.0
            
            # Convert to grayscale and look for bright spots (potential jewelry)
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Count bright pixels (potential jewelry)
            bright_pixels = np.sum(gray_face > 200)
            total_pixels = gray_face.size
            
            return bright_pixels / total_pixels if total_pixels > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_face_visibility(self, landmarks: np.ndarray, frame: np.ndarray) -> float:
        """Calculate overall face visibility"""
        try:
            # Calculate face region
            face_left = int(min(landmarks[:, 0]))
            face_right = int(max(landmarks[:, 0]))
            face_top = int(min(landmarks[:, 1]))
            face_bottom = int(max(landmarks[:, 1]))
            
            # Ensure bounds
            face_left = max(0, face_left)
            face_right = min(frame.shape[1], face_right)
            face_top = max(0, face_top)
            face_bottom = min(frame.shape[0], face_bottom)
            
            if face_bottom <= face_top or face_right <= face_left:
                return 0.0
            
            # Analyze face region for visibility
            face_region = frame[face_top:face_bottom, face_left:face_right]
            if face_region.size == 0:
                return 0.0
            
            # Convert to grayscale and analyze
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate visibility based on contrast and brightness
            mean_brightness = np.mean(gray_face)
            std_dev = np.std(gray_face)
            
            # Normalize to 0-1 range
            brightness_score = min(1.0, mean_brightness / 255.0)
            contrast_score = min(1.0, std_dev / 100.0)
            
            # Combined visibility score
            visibility = (brightness_score + contrast_score) / 2.0
            
            return max(0.0, min(1.0, visibility))
            
        except Exception:
            return 0.0
    
    def _calculate_eye_visibility(self, landmarks: np.ndarray, frame: np.ndarray) -> float:
        """Calculate eye visibility"""
        try:
            left_eye = landmarks[self.LEFT_EYE_START:self.LEFT_EYE_END]
            right_eye = landmarks[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
            
            if len(left_eye) == 0 or len(right_eye) == 0:
                return 0.0
            
            # Calculate visibility for both eyes
            left_visibility = self._calculate_single_eye_visibility(left_eye, frame)
            right_visibility = self._calculate_single_eye_visibility(right_eye, frame)
            
            return (left_visibility + right_visibility) / 2.0
            
        except Exception:
            return 0.0
    
    def _calculate_single_eye_visibility(self, eye_landmarks: np.ndarray, frame: np.ndarray) -> float:
        """Calculate visibility for a single eye"""
        try:
            # Define eye region
            eye_left = int(min(eye_landmarks[:, 0])) - 5
            eye_right = int(max(eye_landmarks[:, 0])) + 5
            eye_top = int(min(eye_landmarks[:, 1])) - 5
            eye_bottom = int(max(eye_landmarks[:, 1])) + 5
            
            # Ensure bounds
            eye_left = max(0, eye_left)
            eye_right = min(frame.shape[1], eye_right)
            eye_top = max(0, eye_top)
            eye_bottom = min(frame.shape[0], eye_bottom)
            
            if eye_bottom <= eye_top or eye_right <= eye_left:
                return 0.0
            
            # Analyze eye region
            eye_region = frame[eye_top:eye_bottom, eye_left:eye_right]
            if eye_region.size == 0:
                return 0.0
            
            # Convert to grayscale and analyze
            gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate visibility based on contrast and brightness
            mean_brightness = np.mean(gray_eye)
            std_dev = np.std(gray_eye)
            
            # Normalize to 0-1 range
            brightness_score = min(1.0, mean_brightness / 255.0)
            contrast_score = min(1.0, std_dev / 100.0)
            
            # Eye visibility score (weighted towards contrast)
            visibility = (brightness_score * 0.3 + contrast_score * 0.7)
            
            return max(0.0, min(1.0, visibility))
            
        except Exception:
            return 0.0
    
    def _calculate_mouth_visibility(self, landmarks: np.ndarray, frame: np.ndarray) -> float:
        """Calculate mouth visibility"""
        try:
            mouth_points = landmarks[48:68]  # mouth landmarks
            
            if len(mouth_points) == 0:
                return 0.0
            
            # Define mouth region
            mouth_left = int(min(mouth_points[:, 0])) - 10
            mouth_right = int(max(mouth_points[:, 0])) + 10
            mouth_top = int(min(mouth_points[:, 1])) - 10
            mouth_bottom = int(max(mouth_points[:, 1])) + 10
            
            # Ensure bounds
            mouth_left = max(0, mouth_left)
            mouth_right = min(frame.shape[1], mouth_right)
            mouth_top = max(0, mouth_top)
            mouth_bottom = min(frame.shape[0], mouth_bottom)
            
            if mouth_bottom <= mouth_top or mouth_right <= mouth_left:
                return 0.0
            
            # Analyze mouth region
            mouth_region = frame[mouth_top:mouth_bottom, mouth_left:mouth_right]
            if mouth_region.size == 0:
                return 0.0
            
            # Convert to grayscale and analyze
            gray_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate visibility based on contrast and brightness
            mean_brightness = np.mean(gray_mouth)
            std_dev = np.std(gray_mouth)
            
            # Normalize to 0-1 range
            brightness_score = min(1.0, mean_brightness / 255.0)
            contrast_score = min(1.0, std_dev / 100.0)
            
            # Mouth visibility score
            visibility = (brightness_score * 0.4 + contrast_score * 0.6)
            
            return max(0.0, min(1.0, visibility))
            
        except Exception:
            return 0.0
    
    def _calculate_nose_visibility(self, landmarks: np.ndarray, frame: np.ndarray) -> float:
        """Calculate nose visibility"""
        try:
            nose_points = landmarks[27:36]  # nose landmarks
            
            if len(nose_points) == 0:
                return 0.0
            
            # Define nose region
            nose_left = int(min(nose_points[:, 0])) - 8
            nose_right = int(max(nose_points[:, 0])) + 8
            nose_top = int(min(nose_points[:, 1])) - 8
            nose_bottom = int(max(nose_points[:, 1])) + 8
            
            # Ensure bounds
            nose_left = max(0, nose_left)
            nose_right = min(frame.shape[1], nose_right)
            nose_top = max(0, nose_top)
            nose_bottom = min(frame.shape[0], nose_bottom)
            
            if nose_bottom <= nose_top or nose_right <= nose_left:
                return 0.0
            
            # Analyze nose region
            nose_region = frame[nose_top:nose_bottom, nose_left:nose_right]
            if nose_region.size == 0:
                return 0.0
            
            # Convert to grayscale and analyze
            gray_nose = cv2.cvtColor(nose_region, cv2.COLOR_BGR2GRAY)
            
            # Calculate visibility based on contrast and brightness
            mean_brightness = np.mean(gray_nose)
            std_dev = np.std(gray_nose)
            
            # Normalize to 0-1 range
            brightness_score = min(1.0, mean_brightness / 255.0)
            contrast_score = min(1.0, std_dev / 100.0)
            
            # Nose visibility score
            visibility = (brightness_score * 0.5 + contrast_score * 0.5)
            
            return max(0.0, min(1.0, visibility))
            
        except Exception:
            return 0.0 