import cv2
import time
import numpy as np
from deepface import DeepFace
import os
from datetime import datetime

class CameraInterface:
    def __init__(self, face_detector, liveness_detector):
        
        self.face_detector = face_detector
        self.liveness_detector = liveness_detector
        self.cap = None
        self.is_running = False
        
        # settings
        self.capture_countdown = 0
        self.capture_countdown_start = 0
        self.COUNTDOWN_DURATION = 3  # seconds
        self.photo_captured = False
        self.captured_frame = None
        self.preprocessed_frame = None  # for quality analysus
        
        self.output_dir = "aproved_img"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # hardware settings
        self.CAMERA_WIDTH = 1280
        self.CAMERA_HEIGHT = 720
        
    def initialize_camera(self, camera_index=0):
        """Initialize camera with error handling and recovery"""
        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            
            # Set camera properties with validation
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            # Warm up camera with multiple reads
            for _ in range(10):
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print(f"Warning: Camera warm-up frame {_} failed")
                    continue
                time.sleep(0.1)  # Small delay between reads
                
            print("✅ Camera initialized successfully")
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            # Try fallback camera
            self._try_fallback_camera()
    
    def _try_fallback_camera(self):
        """Try alternative camera initialization methods"""
        print("🔄 Attempting fallback camera initialization...")
        
        # Try different camera indices
        for camera_index in [1, 2, -1]:
            try:
                if self.cap:
                    self.cap.release()
                
                self.cap = cv2.VideoCapture(camera_index)
                if self.cap.isOpened():
                    # Test frame capture
                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f"✅ Fallback camera {camera_index} initialization successful")
                        # Use default settings for fallback
                        print("Using camera default settings (no custom properties)")
                        return
                    else:
                        self.cap.release()
                        
            except Exception as e:
                print(f"Fallback camera {camera_index} failed: {e}")
                if self.cap:
                    self.cap.release()
        
        # If all fallbacks fail, create a dummy camera
        print("⚠️ All camera initialization attempts failed, creating dummy camera")
        self._create_dummy_camera()
    
    def _create_dummy_camera(self):
        """Create a dummy camera for testing when real camera fails"""
        class DummyCamera:
            def __init__(self):
                self.is_opened = True
                
            def read(self):
                # Return a dummy frame
                dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(dummy_frame, "Camera Error - Press Q to quit", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                return True, dummy_frame
                
            def isOpened(self):
                return self.is_opened
                
            def release(self):
                self.is_opened = False
                
            def set(self, prop, value):
                pass  # Ignore property settings
        
        self.cap = DummyCamera()
        print("✅ Dummy camera created for testing")
    
    def _validate_frame(self, frame):
        """Validate frame before processing to prevent OpenCV errors"""
        if frame is None:
            return False, "Frame is None"
        
        if not isinstance(frame, np.ndarray):
            return False, f"Frame is not numpy array: {type(frame)}"
        
        if frame.size == 0:
            return False, "Frame has zero size"
        
        if len(frame.shape) != 3:
            return False, f"Frame has invalid shape: {frame.shape}"
        
        h, w, c = frame.shape
        if h <= 0 or w <= 0 or c != 3:
            return False, f"Frame has invalid dimensions: {h}x{w}x{c}"
        
        if frame.dtype != np.uint8:
            return False, f"Frame has invalid dtype: {frame.dtype}"
        
        # Check for NaN or infinite values
        if np.any(np.isnan(frame)) or np.any(np.isinf(frame)):
            return False, "Frame contains NaN or infinite values"
        
        return True, "Frame is valid"
    
    def _safe_camera_read(self):
        """Safely read from camera with error handling and recovery"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if not self.cap or not self.cap.isOpened():
                    print("⚠️ Camera not available, attempting recovery...")
                    self._try_fallback_camera()
                    if not self.cap or not self.cap.isOpened():
                        return False, None, "Camera recovery failed"
                
                ret, frame = self.cap.read()
                
                if not ret:
                    print(f"⚠️ Camera read failed (attempt {retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                
                # Validate frame
                is_valid, validation_msg = self._validate_frame(frame)
                if not is_valid:
                    print(f"⚠️ Invalid frame detected: {validation_msg} (attempt {retry_count + 1}/{max_retries})")
                    retry_count += 1
                    time.sleep(0.1)
                    continue
                
                return True, frame, "Frame captured successfully"
                
            except Exception as e:
                print(f"⚠️ Camera read error: {e} (attempt {retry_count + 1}/{max_retries})")
                retry_count += 1
                
                # Try to recover camera
                if retry_count < max_retries:
                    print("Attempting camera recovery...")
                    self._try_fallback_camera()
                    time.sleep(0.5)
        
        print("❌ All camera read attempts failed")
        return False, None, "Camera read failed after all retries"
            
    def release_camera(self):
        if self.cap:
            self.cap.release()
            
    def start_capture_countdown(self):
        #capture countdown
        self.capture_countdown_start = time.time()
        self.capture_countdown = self.COUNTDOWN_DURATION
        
    def update_countdown(self):
        if self.capture_countdown > 0:
            elapsed = time.time() - self.capture_countdown_start
            remaining = self.COUNTDOWN_DURATION - elapsed
            
            if remaining <= 0:
                self.capture_countdown = 0
                return True  
            else:
                self.capture_countdown = int(remaining) + 1
                
        return False
    
    def capture_photo(self, frame, landmarks):
        
        #save photo
        if self.photo_captured:
            return self.captured_frame
            
        # crop - should include shoulders
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lastname_firstname_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        print(f"Image path: {filepath}")
        
        #crop shoulder level 
        cropped_frame = self.crop_to_shoulders(frame, landmarks)
        
        # Save raw cropped image
        cv2.imwrite(filepath, cropped_frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        
        # preprocessed copy for quality analysis
        preprocessed_frame = self.apply_minimal_preprocessing(cropped_frame.copy())
        
        self.photo_captured = True
        self.captured_frame = cropped_frame  
        self.preprocessed_frame = preprocessed_frame  #
        
        print(f"Image saved in: {self.output_dir}")
        
        return preprocessed_frame  # Return preprocessed frame for quality analysis
    
    def crop_to_shoulders(self, frame, landmarks):
        
        #face boundaries
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        face_left = int(min(x_coords))
        face_right = int(max(x_coords))
        face_top = int(min(y_coords))
        face_bottom = int(max(y_coords))
        
        face_width = face_right - face_left
        face_height = face_bottom - face_top
        
        # calculate crop area , include shoulders
        padding_x = int(face_width * 0.5)


        # space above the head, cropped #
        padding_top = int(face_height * 0.6)  
        # space for shoulders, cropped
        padding_bottom = int(face_height * 0.8) 
        
         
        
        #  crop boundaries #
        crop_left = max(0, face_left - padding_x)
        crop_right = min(frame.shape[1], face_right + padding_x)
        crop_top = max(0, face_top - padding_top)
        crop_bottom = min(frame.shape[0], face_bottom + padding_bottom)
        cropped = frame[crop_top:crop_bottom, crop_left:crop_right]

        return cropped
    
    def enhance_image_quality(self, frame, landmarks):

        # crop face region
        x_coords = landmarks[:, 0]
        y_coords = landmarks[:, 1]
        
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        face_width = x_max - x_min
        face_height = y_max - y_min
        padding_x = int(face_width * 0.3)
        padding_y = int(face_height * 0.3)
        
        x_min = max(0, x_min - padding_x)
        x_max = min(frame.shape[1], x_max + padding_x)
        y_min = max(0, y_min - padding_y)
        y_max = min(frame.shape[0], y_max + padding_y)
        
        # crop the face region
        face_crop = frame[y_min:y_max, x_min:x_max]
        
        # enhance cropped image
        enhanced = self.apply_image_enhancements(face_crop)
        
        # passport photo dimensions, width x height
        target_size = (600, 800) 
        enhanced_resized = cv2.resize(enhanced, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        return enhanced_resized
    
    def apply_image_enhancements(self, image):

        # lab color space
        # L = Lightness (how light or dark the color is)
        # a = green-red axis
        # b = blue-yellow axis
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # merge then convert back to BGR
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        #to reduce noise
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        #sharpen image
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        # minimal adjustment of brightness and contrast
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=10)
        
        return enhanced
    
    def apply_minimal_preprocessing(self, image):

    #for analysis#

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        #  minimal flattening/enhancement
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.5)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.05, beta=5)
        
        return enhanced
    
    def draw_countdown(self, frame):
        if self.capture_countdown > 0:
            h, w = frame.shape[:2]
            
            # countdown 
            font_scale = 5
            thickness = 10
            text = str(self.capture_countdown)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            x = (w - text_size[0]) // 2
            y = (h + text_size[1]) // 2
            
            cv2.circle(frame, (w//2, h//2), 120, (0, 0, 0), -1)
            cv2.circle(frame, (w//2, h//2), 120, (0, 255, 0), 5)
            
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (0, 255, 0), thickness)
        
        return frame
    
    def draw_ui_elements(self, frame, faces, landmarks_list, state):
       
        if state.name == 'SHOW_GUIDELINES':
            frame = self.liveness_detector.draw_guidelines(frame)
            
            # real-time compliance checking
            compliance_status = None
            if len(faces) == 1 and len(landmarks_list) == 1:


                # if single face detected - check compliance
                compliance_status = self.face_detector.get_compliance_status(landmarks_list[0], frame)
            
            #update compliance status for automatic progression
            self.liveness_detector.update_compliance_status(compliance_status)
            
            frame = self.liveness_detector.draw_compliance_status(frame, compliance_status)
            return frame
        
        #if in photo review state, draw photo review interface
        if state.name == 'PHOTO_REVIEW':
            frame = self.liveness_detector.draw_photo_review(frame)
            return frame
        
        # set face boundaries and landmarks
        for i, face in enumerate(faces):
            frame = self.face_detector.draw_face_boundary(frame, face)
            if i < len(landmarks_list):
                frame = self.face_detector.draw_landmarks(frame, landmarks_list[i])
        
        frame = self.liveness_detector.draw_face_guide(frame)
        frame = self.liveness_detector.draw_progress(frame)
        
        # Add blink debug visualization when in blink state
        if state.name == 'BLINK' and len(landmarks_list) > 0:
            frame = self.liveness_detector.draw_blink_debug(frame, landmarks_list[0])
        
        frame = self.draw_countdown(frame)
        
        h, w = frame.shape[:2]
        state_color = self.liveness_detector.get_state_color()
        
        cv2.circle(frame, (w - 50, 50), 20, state_color, -1)
        cv2.circle(frame, (w - 50, 50), 20, (255, 255, 255), 2)
        
        mode_text = ""
        cv2.putText(frame, mode_text, (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return frame
    
    def run(self):
        
        #CAMERA LOOP#
        self.is_running = True
        
        try:
            self.initialize_camera()
            
            while self.is_running:
                # Use safe camera read with error handling
                ret, frame, msg = self._safe_camera_read()
                
                if not ret:
                    print(f"Camera read failed: {msg}")
                    # Try to recover camera
                    time.sleep(0.1)
                    continue
                
                # mirroring
                frame = cv2.flip(frame, 1)
                
                # detect faces, landmarks
                faces, gray = self.face_detector.detect_faces(frame)
                landmarks_list = []
                
                for face in faces:
                    landmarks = self.face_detector.get_landmarks(gray, face)
                    landmarks_list.append(landmarks)
                
                # liveness detection
                state = self.liveness_detector.update(frame, faces, landmarks_list)
                
                #compliance checking
                if state.name in ['WAITING_FOR_FACE'] and len(faces) == 1 and len(landmarks_list) == 1:
                    compliance_status = self.face_detector.get_compliance_status(landmarks_list[0], frame)
                    if self.liveness_detector.should_return_to_guidelines(compliance_status):

                        # go back to guidelines if critical violations
                        self.liveness_detector.reset()
                        self.capture_countdown = 0
                        self.photo_captured = False
                        self.captured_frame = None
                        self.preprocessed_frame = None
                
                # if liveness check completed
                if state.name == 'COMPLETED' and self.capture_countdown == 0:
                    self.start_capture_countdown()
                
                if self.capture_countdown > 0:
                    should_capture = self.update_countdown()
                    if should_capture and len(landmarks_list) > 0:
                        captured_photo = self.capture_photo(frame, landmarks_list[0])
                        
                        # preprocessed captured photo quality check
                        quality_issues = self.face_detector.analyze_captured_photo_quality(captured_photo, landmarks_list[0])
                        
                        # clean deleted
                        self.preprocessed_frame = None
                        
                        if quality_issues:
                            print(f"\nQuality Issues Detected")
                            self.liveness_detector.start_photo_review(quality_issues)
                        else:
                            print("\nPHOTO APPROVED - PROCESS COMPLETE")
                            self.liveness_detector.mark_capture_complete()
                
                frame = self.draw_ui_elements(frame, faces, landmarks_list, state)
                
                if state.name == 'PHOTO_REVIEW' and not self.liveness_detector.has_photo_quality_issues():
                    if time.time() - self.liveness_detector.photo_review_start_time > 2:
                        self.liveness_detector.mark_capture_complete()
                
                cv2.imshow('Face Recognition with Liveness Detection', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):

                    # restart the entire process
                    self.liveness_detector.reset()
                    self.capture_countdown = 0
                    self.photo_captured = False
                    self.captured_frame = None
                    self.preprocessed_frame = None
                elif key == ord('a') and state.name == 'PHOTO_REVIEW':
                    self.liveness_detector.mark_capture_complete() 

                
                # auto-restart on failure
                if state.name == 'FAILED':
                    time.sleep(2)  # 2 seconds before restarting , adjust
                    self.liveness_detector.reset()
                    self.capture_countdown = 0
                    self.photo_captured = False
                    self.captured_frame = None
                    self.preprocessed_frame = None
                    
        except Exception as e:
            print(f"Error in camera loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.release_camera()
            cv2.destroyAllWindows()
            
    def stop(self):
        self.is_running = False