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
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        # cam properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        for _ in range(10):
            self.cap.read()
            
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
        print(f"✅ Raw image captured and saved successfully")
        
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
        # space above the head, cropped
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
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
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
        finally:
            self.release_camera()
            cv2.destroyAllWindows()
            
    def stop(self):
        self.is_running = False