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