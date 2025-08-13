import cv2
import time
import numpy as np
from deepface import DeepFace
import os
from datetime import datetime
from typing import Optional, List, Tuple
from face_detector import FaceDetector
from liveness_detector import LivenessDetector

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
        
        # Simple performance optimization
        self.process_every_n_frames = 2
        
        # hardware settings
        self.CAMERA_WIDTH = 1280
        self.CAMERA_HEIGHT = 720
        
    def check_camera_availability(self, camera_index=0):
        """Check if camera is available without opening it (for requirements checking)"""
        try:
            # Quick check without opening camera
            test_cap = cv2.VideoCapture(camera_index)
            if not test_cap.isOpened():
                return False, f"Camera at index {camera_index} is not available"
            
            # Try to read one frame to verify it works
            ret, frame = test_cap.read()
            test_cap.release()
            
            if not ret or frame is None:
                return False, f"Camera at index {camera_index} cannot read frames"
            
            return True, f"Camera at index {camera_index} is available"
            
        except Exception as e:
            return False, f"Error checking camera {camera_index}: {e}"
    
    def initialize_camera_fallback(self, camera_index=0):
        """Fallback camera initialization with minimal settings"""
        try:
            print("Trying fallback camera initialization...")
            
            # Release any existing camera
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
                self.cap = None
            
            # Open camera with minimal settings
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise Exception(f"Could not open camera at index {camera_index}")
            
            # Test single frame read
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                raise Exception("Camera opened but cannot read frames")
            
            # Don't set any properties - use camera defaults
            print("✅ Fallback camera initialization successful")
            print("Using camera default settings (no custom properties)")
            return True
            
        except Exception as e:
            print(f"❌ Fallback camera initialization failed: {e}")
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
                self.cap = None
            return False
    
    def initialize_camera(self, camera_index=0):
        """Initialize camera with better error handling and retry logic"""
        max_retries = 3
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                print(f"Attempting to initialize camera (attempt {attempt + 1}/{max_retries})...")
                
                # Release any existing camera first
                if hasattr(self, 'cap') and self.cap:
                    self.cap.release()
                    self.cap = None
                
                # Try to open camera
                self.cap = cv2.VideoCapture(camera_index)
                if not self.cap.isOpened():
                    raise Exception(f"Could not open camera at index {camera_index}")
                
                # Test if camera can actually read frames - but only read ONE frame
                # Multiple frame reads during initialization can cause OpenCV crashes
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    raise Exception("Camera opened but cannot read frames")
                
                # Validate frame dimensions and data
                if test_frame.size == 0 or test_frame.shape[0] == 0 or test_frame.shape[1] == 0:
                    raise Exception("Camera returned invalid frame dimensions")
                
                print(f"✅ Camera initialized successfully on attempt {attempt + 1}")
                
                # Set camera properties - but be more conservative
                try:
                    # Set basic properties one by one with error checking
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_WIDTH)
                    time.sleep(0.1)  # Small delay between property changes
                    
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_HEIGHT)
                    time.sleep(0.1)
                    
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    time.sleep(0.1)
                    
                    # Only set autofocus if supported
                    try:
                        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                    except:
                        print("Autofocus not supported by this camera")
                    
                    print("Camera properties set successfully")
                    
                except Exception as prop_error:
                    print(f"Warning: Some camera properties could not be set: {prop_error}")
                    print("Continuing with default camera settings...")
                
                # DON'T warm up camera by reading multiple frames - this causes crashes!
                # Just verify we can still read one more frame after setting properties
                try:
                    ret, verify_frame = self.cap.read()
                    if ret and verify_frame is not None and verify_frame.size > 0:
                        print("Camera verification successful - ready to use")
                    else:
                        print("Warning: Camera verification frame was invalid, but continuing...")
                except Exception as verify_error:
                    print(f"Warning: Camera verification failed: {verify_error}")
                    print("Continuing anyway - camera might work during normal operation...")
                
                return
                
            except Exception as e:
                print(f"❌ Camera initialization attempt {attempt + 1} failed: {e}")
                
                # Release failed camera
                if hasattr(self, 'cap') and self.cap:
                    self.cap.release()
                    self.cap = None
                
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print("All camera initialization attempts failed")
                    # Provide specific error messages for common issues
                    if "Could not open camera" in str(e):
                        print("\n🔍 TROUBLESHOOTING:")
                        print("• Camera index might be wrong - try: python main.py --camera 1")
                        print("• Camera might be in use by another application")
                        print("• Check Windows privacy settings for camera access")
                    elif "cannot read frames" in str(e):
                        print("\n🔍 TROUBLESHOOTING:")
                        print("• Camera driver might be corrupted")
                        print("• Try restarting your computer")
                        print("• Check Device Manager for camera issues")
                    elif "matrix.cpp" in str(e) or "_step >= minstep" in str(e):
                        print("\n🔍 TROUBLESHOOTING:")
                        print("• OpenCV compatibility issue with camera driver")
                        print("• Try updating camera drivers")
                        print("• Try different camera index: python main.py --camera 1")
                        print("• This is a known OpenCV 4.8.1 issue with some cameras")
                    
                    raise Exception(f"Camera initialization failed after {max_retries} attempts: {e}")
        
        raise Exception("Camera initialization failed - unknown error")
            
    def release_camera(self):
        try:
            if hasattr(self, 'cap') and self.cap:
                if self.cap.isOpened():
                    self.cap.release()
                    print("Camera released successfully")
                self.cap = None
        except Exception as e:
            print(f"Error releasing camera: {e}")
        finally:
            # Ensure camera is released
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
                self.cap = None
            
    def start_capture_countdown(self):
        #capture countdown
        print("Starting capture countdown in 1 second...")
        time.sleep(1)  # Give user 1 second to prepare
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
                # Fix: Don't add +1, just use the remaining time
                # This ensures countdown shows correct values: 3, 2, 1, 0
                self.capture_countdown = max(1, int(remaining))
                
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
        
        return preprocessed_frame  
    
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
        # Safety check: ensure frame is not None
        if frame is None:
            print("Warning: draw_countdown called with None frame")
            # Return a fallback frame instead of None
            fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(fallback_frame, "Camera Error - Press Q to quit", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return fallback_frame
            
        if self.capture_countdown > 0:
            h, w = frame.shape[:2]
            ui_scale = self.get_ui_scale_factor(frame)
            
            # countdown with much better scaling - not too large
            font_scale = 1.5 * ui_scale  # Reduced from 2.5 to 1.5
            thickness = max(1, int(2 * ui_scale))  # Reduced from 3 to 2
            text = str(self.capture_countdown)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            x = (w - text_size[0]) // 2
            y = (h + text_size[1]) // 2
            
            circle_radius = int(60 * ui_scale)  # Reduced from 80 to 60
            circle_thickness = max(1, int(2 * ui_scale))  # Reduced from 3 to 2
            
            # Draw black circle background
            cv2.circle(frame, (w//2, h//2), circle_radius, (0, 0, 0), -1)
            cv2.circle(frame, (w//2, h//2), circle_radius, (0, 255, 0), circle_thickness)
            
            # Draw countdown number
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (0, 255, 0), thickness)
            
            # Add "GET READY" message above countdown
            ready_text = "GET READY!"
            ready_font_scale = 0.8 * ui_scale  # Reduced from 1.0 to 0.8
            ready_thickness = max(1, int(1 * ui_scale))  # Reduced from 2 to 1
            ready_text_size = cv2.getTextSize(ready_text, cv2.FONT_HERSHEY_SIMPLEX, ready_font_scale, ready_thickness)[0]
            ready_x = (w - ready_text_size[0]) // 2
            ready_y = y - 40  # Reduced from 60 to 40
            
            cv2.putText(frame, ready_text, (ready_x, ready_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       ready_font_scale, (255, 255, 255), ready_thickness)
        elif self.photo_captured:
            # Show completion message instead of countdown
            h, w = frame.shape[:2]
            ui_scale = self.get_ui_scale_factor(frame)
            
            # Draw completion message with much better scaling
            completion_text = "PHOTO CAPTURED!"
            font_scale = 1.0 * ui_scale  # Reduced from 1.5 to 1.0
            thickness = max(1, int(1 * ui_scale))  # Reduced from 2 to 1
            text_size = cv2.getTextSize(completion_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            
            x = (w - text_size[0]) // 2
            y = (h + text_size[1]) // 2
            
            circle_radius = int(60 * ui_scale)  # Reduced from 80 to 60
            circle_thickness = max(1, int(2 * ui_scale))  # Reduced from 3 to 2
            cv2.circle(frame, (w//2, h//2), circle_radius, (0, 0, 0), -1)
            cv2.circle(frame, (w//2, h//2), circle_radius, (0, 255, 0), circle_thickness)
            
            # Draw completion text
            cv2.putText(frame, completion_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (0, 255, 0), thickness)
            
            # Add success message below
            success_text = "Quality assessment completed!"
            success_font_scale = 0.6 * ui_scale  # Reduced from 0.8 to 0.6
            success_thickness = max(1, int(1 * ui_scale))  # Keep at 1
            success_text_size = cv2.getTextSize(success_text, cv2.FONT_HERSHEY_SIMPLEX, success_font_scale, success_thickness)[0]
            success_x = (w - success_text_size[0]) // 2
            success_y = y + 30  # Reduced from 50 to 30
            
            cv2.putText(frame, success_text, (success_x, success_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       success_font_scale, (255, 255, 255), success_thickness)
        
        return frame
    
    def get_ui_scale_factor(self, frame):
        """Much smaller text scaling - keep text appropriately sized for different resolutions"""
        # Safety check: ensure frame is not None
        if frame is None:
            print("Warning: get_ui_scale_factor called with None frame")
            return 0.08  # Return much smaller default scale
            
        h, w = frame.shape[:2]
        
        # Much smaller scaling based on resolution
        if w > 1400:  # Fullscreen mode (1920x1080, 2560x1440, etc.)
            return 0.08  # Much smaller scale for high resolution
        elif w > 1000:  # Medium resolution
            return 0.12
        else:  # Low resolution
            return 0.15
    
    def get_display_scale_factor(self):
        """Get the current display scale factor for coordinate scaling"""
        try:
            if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
                # Get the actual frame size
                frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Get the window size (if available)
                # Note: OpenCV doesn't provide direct access to window size
                # So we'll use a reasonable default scale
                if frame_width > 0 and frame_height > 0:
                    # Assume display is scaled to fit screen
                    # This is a reasonable approximation
                    return 1.0
                else:
                    return 1.0
            else:
                return 1.0
        except Exception as e:
            print(f"Error getting display scale: {e}")
            return 1.0
    
    def scale_frame_for_display(self, frame):
        """Simple scaling - no frame scaling, just better text sizing"""
        return frame  # Keep original frame size
    
    def draw_ui_elements(self, frame, faces, landmarks_list, state):
        # Safety check: ensure frame is not None
        if frame is None:
            print("Warning: draw_ui_elements called with None frame")
            # Return a fallback frame instead of None
            fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(fallback_frame, "Camera Error - Press Q to quit", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return fallback_frame
            
        # Get adaptive scaling for UI elements
        ui_scale = self.get_ui_scale_factor(frame)
        
        # Safety check: ensure state is not None and has a name attribute
        if state is None:
            print("Warning: draw_ui_elements called with None state")
            return frame
            
        # Safety check: ensure state has a name attribute
        if not hasattr(state, 'name'):
            print("Warning: draw_ui_elements called with state that has no name attribute")
            return frame
            
        # Safety check: ensure faces and landmarks_list are not None
        if faces is None:
            faces = []
        if landmarks_list is None:
            landmarks_list = []
       
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
            if frame is not None:
                # Get display scale factor for proper coordinate scaling
                display_scale = self.get_display_scale_factor()
                result = self.face_detector.draw_face_boundary(frame, face, display_scale)
                frame = result if result is not None else frame
            if i < len(landmarks_list) and frame is not None:
                # Get display scale factor for proper coordinate scaling
                display_scale = self.get_display_scale_factor()
                result = self.face_detector.draw_landmarks(frame, landmarks_list[i], display_scale)
                frame = result if result is not None else frame
        
        if frame is not None:
            result = self.liveness_detector.draw_face_guide(frame)
            frame = result if result is not None else frame
        
        if frame is not None:
            result = self.liveness_detector.draw_progress(frame)
            frame = result if result is not None else frame
        
        # Add head pose debug display when in liveness detection states
        if (frame is not None and state.name in ['LOOK_LEFT', 'LOOK_RIGHT'] and 
            landmarks_list and len(landmarks_list) > 0):
            result = self.liveness_detector.draw_head_pose_debug(frame, landmarks_list[0])
            frame = result if result is not None else frame
        
        if frame is not None:
            result = self.draw_countdown(frame)
            frame = result if result is not None else frame
        
        # Final safety check before accessing frame.shape
        if frame is None:
            print("Warning: frame is None after all drawing operations, creating fallback")
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera Error - Press Q to quit", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        h, w = frame.shape[:2]
        state_color = self.liveness_detector.get_state_color()
        
        # Scale UI elements based on screen size
        circle_radius = int(15 * ui_scale)  # Reduced from 20 to 15
        circle_x = w - int(40 * ui_scale)  # Reduced from 50 to 40
        circle_y = int(40 * ui_scale)  # Reduced from 50 to 40
        
        cv2.circle(frame, (circle_x, circle_y), circle_radius, state_color, -1)
        cv2.circle(frame, (circle_x, circle_y), circle_radius, (255, 255, 255), max(1, int(1 * ui_scale)))  # Reduced from 2 to 1
        
        # Simple controls text like reference file
        if w > 1400:  # Fullscreen mode
            controls_text = "Q: Quit  |  R: Restart  |  F: Exit Fullscreen"
            cv2.putText(frame, controls_text, (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)  # Reduced from 0.5 to 0.4
        
        mode_text = ""
        font_scale = 0.4 * ui_scale  # Reduced from 0.5 to 0.4
        text_thickness = max(1, int(1 * ui_scale))
        text_y = h - int(50 * ui_scale) if w > 1400 else h - int(20 * ui_scale)  # Reduced from 60 to 50
        
        cv2.putText(frame, mode_text, (int(10 * ui_scale), text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), text_thickness)
        
        return frame
    
    def run(self, camera_index=0):
        
        #CAMERA LOOP#
        self.is_running = True
        
        # Loop safety: prevent infinite loops
        consecutive_errors = 0
        max_consecutive_errors = 10
        consecutive_none_frames = 0
        max_consecutive_none_frames = 5
        camera_crash_count = 0
        max_camera_crashes = 3
        
        try:
            # Try to initialize camera first
            camera_initialized = False
            
            try:
                self.initialize_camera(camera_index)
                print("✅ Camera initialized successfully!")
                camera_initialized = True
            except Exception as e:
                print(f"❌ Main camera initialization failed: {e}")
                
                # Try fallback initialization
                print("Attempting fallback camera initialization...")
                if self.initialize_camera_fallback(camera_index):
                    print("✅ Fallback camera initialization successful!")
                    camera_initialized = True
                else:
                    print("❌ Both main and fallback camera initialization failed")
                    print("\n" + "="*60)
                    print("CAMERA INITIALIZATION FAILED")
                    print("="*60)
                    print("Common causes and solutions:")
                    print("1. Camera is in use by another application (Zoom, Teams, etc.)")
                    print("   → Close other camera applications and try again")
                    print("2. Camera permissions not granted")
                    print("   → Check Windows privacy settings for camera access")
                    print("3. Camera driver issues")
                    print("   → Try restarting your computer")
                    print("4. Wrong camera index")
                    print("   → Try different camera index: python main.py --camera 1")
                    print("5. Skip camera check: python main.py --no-check")
                    print("6. OpenCV compatibility issue")
                    print("   → Try updating camera drivers")
                    print("   → This is a known OpenCV 4.8.1 issue with some cameras")
                    print("="*60)
                    
                    # Don't continue if camera initialization fails
                    print("Exiting due to camera initialization failure")
                    return
            
            # Verify camera is properly initialized before proceeding
            if not camera_initialized or self.cap is None or not self.cap.isOpened():
                print("❌ Camera object is invalid after initialization")
                return
            
            # Create fullscreen window by default
            window_name = 'Face Recognition with Liveness Detection'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # WINDOW_NORMAL allows resizing
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print("Camera window created in fullscreen mode")
            print("Press 'F' to exit fullscreen, 'ESC' to exit fullscreen, 'Q' to quit")
            
            frame_count = 0
            process_every_n_frames = 3  # Process every 3rd frame for better performance (was 2)
            is_fullscreen = True  # Start in fullscreen mode
            
            # Performance optimization: skip frames for heavy operations
            last_face_detection_time = 0
            face_detection_interval = 0.1  # Detect faces every 100ms (10 FPS for detection)
            
            while self.is_running:
                try:
                    # Safety check: ensure camera is still valid
                    if self.cap is None or not self.cap.isOpened():
                        print("❌ Camera became invalid during operation")
                        break
                    
                    # ROBUST FRAME READING WITH ERROR RECOVERY
                    frame = None
                    ret = False
                    
                    try:
                        ret, frame = self.cap.read()
                    except cv2.error as cv_error:
                        # Handle OpenCV-specific errors (like matrix assertion failures)
                        if "matrix.cpp" in str(cv_error) or "_step >= minstep" in str(cv_error):
                            print(f"⚠️  OpenCV matrix error detected: {cv_error}")
                            camera_crash_count += 1
                            
                            if camera_crash_count <= max_camera_crashes:
                                print(f"Attempting camera recovery (attempt {camera_crash_count}/{max_camera_crashes})...")
                                
                                # Try to reinitialize camera
                                try:
                                    self.release_camera()
                                    time.sleep(1)  # Wait for camera to reset
                                    
                                    if self.initialize_camera_fallback(camera_index):
                                        print("✅ Camera recovered successfully!")
                                        camera_crash_count = 0  # Reset crash counter
                                        continue  # Skip to next frame
                                    else:
                                        print("❌ Camera recovery failed")
                                        break
                                except Exception as recovery_error:
                                    print(f"❌ Camera recovery failed: {recovery_error}")
                                    break
                            else:
                                print(f"❌ Too many camera crashes ({camera_crash_count}), giving up")
                                break
                        else:
                            # Re-raise non-matrix OpenCV errors
                            raise cv_error
                    except Exception as e:
                        print(f"❌ Unexpected error reading frame: {e}")
                        consecutive_errors += 1
                        
                        if consecutive_errors >= max_consecutive_errors:
                            print(f"Too many consecutive errors ({consecutive_errors}), breaking out of loop")
                            break
                        
                        time.sleep(0.1)
                        continue
                    
                    if not ret or frame is None:
                        print("Failed to capture frame")
                        consecutive_errors += 1
                        
                        # Break out if too many consecutive camera failures
                        if consecutive_errors >= max_consecutive_errors:
                            print(f"Too many consecutive camera failures ({consecutive_errors}), breaking out of loop")
                            break
                        
                        # Small delay before retrying
                        time.sleep(0.1)
                        continue
                    
                    # Reset error counters on successful frame read
                    consecutive_errors = 0
                    camera_crash_count = 0
                    
                    # Check if we should stop (for external shutdown requests)
                    if not self.is_running:
                        print("Shutdown requested - stopping camera loop...")
                        break
                    
                    # mirroring
                    frame = cv2.flip(frame, 1)
                    
                    frame_count += 1
                    current_time = time.time()
                    
                    # Initialize variables for this frame iteration
                    faces = getattr(self, '_last_faces', [])
                    landmarks_list = getattr(self, '_last_landmarks', [])
                    state = getattr(self, '_last_state', None)
                    
                    # Safety check: if cached values are consistently invalid, reset them
                    if (faces is None or landmarks_list is None or state is None or 
                        (state and not hasattr(state, 'name'))):
                        print("Warning: Invalid cached values detected, resetting to defaults")
                        faces = []
                        landmarks_list = []
                        state = self.liveness_detector.state
                        # Clear invalid cache
                        self._last_faces = faces
                        self._last_landmarks = landmarks_list
                        self._last_state = state
                    
                    # Only process face detection at specified intervals for performance
                    should_detect_faces = (current_time - last_face_detection_time) >= face_detection_interval
                    
                    if should_detect_faces:
                        try:
                            # Simple face detection like reference file
                            try:
                                faces, gray = self.face_detector.detect_faces(frame)
                            except Exception as e:
                                print(f"Error in face detection: {e}")
                                # Use empty results on face detection failure
                                faces = []
                                gray = frame
                            
                            landmarks_list = []
                            
                            for face in faces:
                                try:
                                    landmarks = self.face_detector.get_landmarks(gray, face)
                                    landmarks_list.append(landmarks)
                                except Exception as e:
                                    print(f"Error in landmark extraction: {e}")
                                    continue
                            
                            # liveness detection with error handling
                            try:
                                state = self.liveness_detector.update(frame, faces, landmarks_list)
                            except Exception as e:
                                print(f"Error in liveness detection: {e}")
                                # Use a safe default state
                                state = self.liveness_detector.state
                            
                            #compliance checking
                            if (state and hasattr(state, 'name') and state.name in ['WAITING_FOR_FACE'] and 
                                faces and landmarks_list and 
                                len(faces) == 1 and len(landmarks_list) == 1):
                                try:
                                    if landmarks_list and len(landmarks_list) > 0:
                                        compliance_status = self.face_detector.get_compliance_status(landmarks_list[0], frame)
                                    else:
                                        compliance_status = None
                                    if self.liveness_detector.should_return_to_guidelines(compliance_status):

                                        # go back to guidelines if critical violations
                                        self.liveness_detector.reset()
                                        self.capture_countdown = 0
                                        self.photo_captured = False
                                        self.captured_frame = None
                                        self.preprocessed_frame = None
                                except Exception as e:
                                    print(f"Error in compliance checking: {e}")
                                    continue
                            
                            # if liveness check completed and photo not yet captured
                            if (state and hasattr(state, 'name') and 
                                state.name == 'COMPLETED' and 
                                self.capture_countdown == 0 and 
                                not self.photo_captured):
                                print("🎯 Starting photo capture countdown...")
                                self.start_capture_countdown()
                            elif (state and hasattr(state, 'name') and 
                                  state.name == 'COMPLETED' and 
                                  self.photo_captured):
                                # Photo already captured, move to completion state
                                print("📸 Photo captured! Moving to completion state...")
                                self.liveness_detector.mark_capture_complete()
                            elif (state and hasattr(state, 'name') and 
                                  state.name in ['CAPTURE_COMPLETE', 'PHOTO_REVIEW']):
                                # Already in completion state, don't restart countdown
                                pass
                            
                            last_face_detection_time = current_time
                            
                        except Exception as e:
                            print(f"Error in face detection/liveness processing: {e}")
                            # Use cached results on error
                            faces = getattr(self, '_last_faces', [])
                            landmarks_list = getattr(self, '_last_landmarks', [])
                            state = getattr(self, '_last_state', None)
                    
                    # Cache results for frame skipping
                    # Ensure we have valid values before caching
                    if faces is None:
                        faces = []
                    if landmarks_list is None:
                        landmarks_list = []
                    if state is None:
                        state = self.liveness_detector.state
                        
                    # Ensure state has a name attribute (only check once)
                    if state and not hasattr(state, 'name'):
                        print("Warning: state object has no name attribute, using default state")
                        state = self.liveness_detector.state
                        
                    self._last_faces = faces
                    self._last_landmarks = landmarks_list
                    self._last_state = state
                    
                    # 🎯 CAPTURE PHOTO FIRST (before UI drawing to avoid overlays)
                    # This ensures the saved photo is clean without text, circles, or UI elements
                    if self.capture_countdown > 0:
                        try:
                            self.update_countdown()
                            if self.capture_countdown == 0 and not self.photo_captured:
                                # Ensure we have valid landmarks before accessing
                                try:
                                    if landmarks_list and len(landmarks_list) > 0:
                                        # Capture photo BEFORE UI drawing to get clean image
                                        captured_photo = self.capture_photo(frame, landmarks_list[0])
                                        print("✅ Photo captured successfully! Starting quality check...")
                                        
                                        # 🔍 QUALITY CHECK - Analyze captured photo
                                        try:
                                            quality_issues = self.face_detector.analyze_captured_photo_quality(captured_photo, landmarks_list[0])
                                            
                                            if quality_issues:
                                                print(f"⚠️  Quality Issues Detected: {len(quality_issues)} issues")
                                                self.liveness_detector.start_photo_review(quality_issues)
                                            else:
                                                print("✅ PHOTO QUALITY APPROVED - PROCESS COMPLETE")
                                                self.liveness_detector.mark_capture_complete()
                                                
                                        except Exception as e:
                                            print(f"Error in quality check: {e}")
                                            # Continue without quality check if it fails
                                            self.liveness_detector.mark_capture_complete()
                                        
                                    else:
                                        self.capture_photo(frame, None)
                                        print("✅ Photo captured successfully! Moving to completion...")
                                except Exception as e:
                                    print(f"Error in photo capture: {e}")
                                    continue
                        except Exception as e:
                            print(f"Error in countdown update: {e}")
                            continue
                    
                    # Always draw UI elements AFTER photo capture (lightweight operation)
                    try:
                        frame = self.draw_ui_elements(frame, faces, landmarks_list, state)
                    except Exception as e:
                        print(f"Error in draw_ui_elements: {e}")
                        # Create a simple fallback frame
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, "UI Drawing Error - Press Q to quit", (50, 240), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Safety check: ensure frame is not None after UI drawing
                    if frame is None:
                        print("Warning: draw_ui_elements returned None frame, using fallback frame")
                        consecutive_none_frames += 1
                        
                        # Break out if too many consecutive None frames to prevent infinite loops
                        if consecutive_none_frames >= max_consecutive_none_frames:
                            print(f"Too many consecutive None frames ({consecutive_none_frames}), breaking out of loop")
                            break
                        
                        # Create a simple fallback frame to prevent infinite loops
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, "Camera Error - Press Q to quit", (50, 240), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    else:
                        # Reset None frame counter on successful frame
                        consecutive_none_frames = 0
                    
                    # 🔍 AUTO-PROCEED FROM PHOTO REVIEW (after quality check)
                    if (state and hasattr(state, 'name') and 
                        state.name == 'PHOTO_REVIEW' and 
                        not self.liveness_detector.has_photo_quality_issues()):
                        # Quality check completed successfully, auto-proceed
                        if time.time() - self.liveness_detector.photo_review_start_time > 3:  # 3 seconds delay
                            print("✅ Quality check completed - Auto-proceeding to completion...")
                            self.liveness_detector.mark_capture_complete()
                    

                    
                    # Display frame
                    try:
                        # Add keyboard shortcuts help text to frame
                        if frame is not None:
                            h, w = frame.shape[:2]
                            help_text = [
                                "Q: Quit | R: Reset | M: Manual Advance | S: Skip to Complete | A: Approve Photo"
                            ]
                            for i, text in enumerate(help_text):
                                y_pos = h - 20 - (len(help_text) - 1 - i) * 20
                                cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)  # Reduced from 0.5 to 0.4
                        
                        cv2.imshow(window_name, frame)
                    except Exception as e:
                        print(f"Error in frame display: {e}")
                        continue
                    
                    # Handle key events
                    try:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                            break
                        elif key == ord('f') or key == ord('F'):
                            if is_fullscreen:
                                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                                is_fullscreen = False
                                print("Exited fullscreen mode")
                            else:
                                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                                is_fullscreen = True
                                print("Entered fullscreen mode")
                        elif key == ord('r') or key == ord('R'):
                            self.liveness_detector.reset()
                            self.capture_countdown = 0
                            self.photo_captured = False
                            print("Reset liveness detection")
                        elif key == ord('m') or key == ord('M'):
                            # Manual advance through liveness steps
                            self.liveness_detector.manual_advance()
                            print("Manual advance triggered")
                        elif key == ord('s') or key == ord('S'):
                            # Skip to completed state (emergency override)
                            if hasattr(self.liveness_detector, 'state'):
                                print(f"Current state: {self.liveness_detector.state.name}")
                                if self.liveness_detector.state.name in ['LOOK_LEFT', 'LOOK_RIGHT', 'BLINK']:
                                    self.liveness_detector.state = self.liveness_detector.LivenessState.COMPLETED
                                    print("EMERGENCY: Skipped to COMPLETED state")
                        elif key == ord('a') or key == ord('A'):
                            # Approve photo and proceed (when in photo review)
                            if (hasattr(self.liveness_detector, 'state') and 
                                self.liveness_detector.state.name == 'PHOTO_REVIEW'):
                                print("✅ Manual approval - Proceeding to completion...")
                                self.liveness_detector.mark_capture_complete()
                            else:
                                print("Manual approval only available during photo review")
                    except Exception as e:
                        print(f"Error in key handling: {e}")
                        continue
                    
                    # Small delay to maintain reasonable frame rate
                    time.sleep(0.01)  # 10ms delay for ~100 FPS display
                    
                except Exception as e:
                    print(f"Error in camera loop: {e}")
                    import traceback
                    traceback.print_exc()
                    consecutive_errors += 1
                    
                    # Break out if too many consecutive errors to prevent infinite loops
                    if consecutive_errors >= max_consecutive_errors:
                        print(f"Too many consecutive errors ({consecutive_errors}), breaking out of loop")
                        break
                    
                    # Small delay before retrying to prevent rapid error loops
                    time.sleep(0.1)
                    continue
                    
        except Exception as e:
            print(f"Error in camera loop: {e}")
        finally:
            print("Cleaning up camera resources...")
            self.release_camera()
            cv2.destroyAllWindows()
            print("Camera cleanup completed")
            
    def stop(self):
        print("Stopping camera interface...")
        self.is_running = False
        # Force cleanup if stop is called externally
        self.release_camera()
        cv2.destroyAllWindows()