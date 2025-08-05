import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
from typing import Tuple, List, Optional, Dict, Any
from base_detector import BaseDetector
from config_manager import ConfigManager

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
    
    def _initialize_components(self) -> None:
        face_config = self.config_manager.get_face_detection_config()
        eye_config = self.config_manager.get_eye_detection_config()
        head_config = self.config_manager.get_head_pose_config()
        
        self.landmarks_file = face_config.get('landmarks_file', 'shape_predictor_68_face_landmarks.dat')
        
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.landmarks_file)
        
        self.eye_indices = {
            'left_eye_start': eye_config.get('left_eye_start', 36),
            'left_eye_end': eye_config.get('left_eye_end', 42),
            'right_eye_start': eye_config.get('right_eye_start', 42),
            'right_eye_end': eye_config.get('right_eye_end', 48)
        }
        
        # head pose model points
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             #nose tip
            (0.0, -330.0, -65.0),        #chin
            (-225.0, 170.0, -135.0),     #left eye left corner
            (225.0, 170.0, -135.0),      #right eye right corner
            (-150.0, -150.0, -125.0),    #left mouth corner
            (150.0, -150.0, -125.0)      #right mouth corner
        ])
        
        self.logger.info("Face detection components initialized")
    
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
    
    #liveness - blinking behavior
    def _analyze_blinking(self, landmarks: np.ndarray) -> Dict[str, Any]:
        left_eye = landmarks[self.eye_indices['left_eye_start']:self.eye_indices['left_eye_end']]
        right_eye = landmarks[self.eye_indices['right_eye_start']:self.eye_indices['right_eye_end']]
        
        left_ear = self._calculate_ear(left_eye)
        right_ear = self._calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        ear_threshold = self.config_manager.get('eye_detection.ear_threshold', 0.25)
        
        return {
            'is_blinking': avg_ear < ear_threshold,
            'left_ear': left_ear,
            'right_ear': right_ear,
            'avg_ear': avg_ear,
            'threshold': ear_threshold
        }
    
    #calculate ear
    def _calculate_ear(self, eye: np.ndarray) -> float:

        #vertical distances
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        
        #horizontal distance
        C = dist.euclidean(eye[0], eye[3])
        
        ear = (A + B) / (2.0 * C)
        return ear
    
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

        coverage_config = self.config_manager.get('face_coverage', {})
        
        left_eye = landmarks[self.eye_indices['left_eye_start']:self.eye_indices['left_eye_end']]
        right_eye = landmarks[self.eye_indices['right_eye_start']:self.eye_indices['right_eye_end']]
        
        eye_coverage = 0.0  # Placeholder for actual calculation
        
        return {
            'has_glasses': eye_coverage > coverage_config.get('max_glasses_coverage', 0.3),
            'coverage_ratio': eye_coverage
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
    
    def cleanup(self) -> None:
        self.detector = None
        self.predictor = None
        self._last_detection = None
        self._last_landmarks = None
        super().cleanup() 