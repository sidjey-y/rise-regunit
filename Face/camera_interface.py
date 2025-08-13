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
        
        self.output_dir = "aproved_img"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Simple performance optimization
        self.process_every_n_frames = 2
        
        # hardware settings
        self.CAMERA_WIDTH = 1280
        self.CAMERA_HEIGHT = 720
        
    def initialize_camera(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("Could not open camera")
        
        # cam properties,
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        for _ in range(10):
            self.cap.read()
            
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
        
        # Check if photo already captured
        if self.photo_captured:
            return self.captured_frame
            
        # crop - should include shoulders
        cropped_frame = self.crop_to_shoulders(frame, landmarks)
        
        # Store the cropped frame in memory (don't save yet)
        self.photo_captured = True
        self.captured_frame = cropped_frame
        
        print(f"✅ Photo captured and stored in memory - awaiting quality check")
        
        return cropped_frame
    
    def save_captured_photo(self) -> str:
        """Save the captured photo to disk after quality checks pass"""
        if not self.photo_captured or self.captured_frame is None:
            print("❌ No photo captured to save")
            return ""
            
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"lastname_firstname_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            # Save the raw cropped image at high quality
            cv2.imwrite(filepath, self.captured_frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            print(f"✅ Photo saved successfully: {filepath}")
            return filepath
        except Exception as e:
            print(f"❌ Error saving photo: {e}")
            return ""
    
    def is_photo_saved(self) -> bool:
        """Check if the captured photo has been saved to disk"""
        if not self.photo_captured:
            return False
        # Check if we have a saved file path (you could store this in an instance variable)
        # For now, we'll assume if photo_captured is True, it means it's ready to be saved
        return True  
    
    def reset_photo_capture(self):
        """Reset photo capture state - useful for retaking photos"""
        self.photo_captured = False
        self.captured_frame = None
        print("🔄 Photo capture state reset - ready for new capture")
    
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
            self._draw_countdown_ui(frame)
        elif self.photo_captured:
            self._draw_completion_ui(frame)
        
        return frame
    
    def _draw_countdown_ui(self, frame):
        """Draw countdown UI elements efficiently"""
        h, w = frame.shape[:2]
        ui_scale = self.get_ui_scale_factor(frame)
        
        # Pre-calculate common values
        center_x, center_y = w // 2, h // 2
        circle_radius = int(120 * ui_scale)
        circle_thickness = max(1, int(5 * ui_scale))
        
        # Draw circle background once
        cv2.circle(frame, (center_x, center_y), circle_radius, (0, 0, 0), -1)
        cv2.circle(frame, (center_x, center_y), circle_radius, (0, 255, 0), circle_thickness)
        
        # Draw countdown number
        countdown_text = str(self.capture_countdown)
        self._draw_centered_text(frame, countdown_text, center_x, center_y, 
                               5 * ui_scale, max(1, int(10 * ui_scale)), (0, 255, 0))
        
        # Draw "GET READY" message above countdown
        self._draw_centered_text(frame, "GET READY!", center_x, center_y - 80, 
                               1.5 * ui_scale, max(1, int(3 * ui_scale)), (255, 255, 255))
    
    def _draw_completion_ui(self, frame):
        """Draw completion UI elements efficiently"""
        h, w = frame.shape[:2]
        ui_scale = self.get_ui_scale_factor(frame)
        
        # Pre-calculate common values
        center_x, center_y = w // 2, h // 2
        circle_radius = int(120 * ui_scale)
        circle_thickness = max(1, int(5 * ui_scale))
        
        # Draw circle background once
        cv2.circle(frame, (center_x, center_y), circle_radius, (0, 0, 0), -1)
        cv2.circle(frame, (center_x, center_y), circle_radius, (0, 255, 0), circle_thickness)
            
            # Draw completion text
        self._draw_centered_text(frame, "PHOTO CAPTURED!", center_x, center_y, 
                               2 * ui_scale, max(1, int(4 * ui_scale)), (0, 255, 0))
        
        # Draw success message below
        self._draw_centered_text(frame, "Quality assessment completed!", center_x, center_y + 60, 
                               1 * ui_scale, max(1, int(2 * ui_scale)), (255, 255, 255))
    
    def _draw_centered_text(self, frame, text, center_x, center_y, font_scale, thickness, color):
        """Helper method to draw centered text efficiently"""
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        x = center_x - text_size[0] // 2
        y = center_y + text_size[1] // 2
        
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, thickness)
    
    def get_ui_scale_factor(self, frame):
        """Simple text scaling - keep text small and readable"""
        # Safety check: ensure frame is not None
        if frame is None:
            print("Warning: get_ui_scale_factor called with None frame")
            return 0.5  # Return default scale
            
        h, w = frame.shape[:2]
        # Always use small, fixed scaling for readable text
        if w > 1400:  # Fullscreen mode
            return 0.5  # Small fixed scale for fullscreen
        else:
            return 0.7  # Slightly larger for windowed mode
    
    def scale_frame_for_display(self, frame):
        """Simple scaling - no frame scaling, just better text sizing"""
        return frame  # Keep original frame size
    
    def draw_ui_elements(self, frame, faces, landmarks_list, state):
        """Draw UI elements efficiently with proper error handling"""
        # Input validation
        if not self._validate_ui_inputs(frame, state, faces, landmarks_list):
            return self._create_fallback_frame()
        
        # Handle state-specific UI first
        if state.name == 'SHOW_GUIDELINES':
            return self._draw_guidelines_state(frame, faces, landmarks_list)
        elif state.name == 'PHOTO_REVIEW':
            return self._draw_photo_review_state(frame)
        
        # Draw common UI elements
        frame = self._draw_common_ui_elements(frame, faces, landmarks_list, state)
        
        # Draw state indicator and controls
        frame = self._draw_state_indicator_and_controls(frame, state)
        
        return frame
    
    def _validate_ui_inputs(self, frame, state, faces, landmarks_list):
        """Validate all inputs for UI drawing"""
        if frame is None:
            print("Warning: draw_ui_elements called with None frame")
            return False
        
        if state is None or not hasattr(state, 'name'):
            print("Warning: Invalid state object")
            return False
        
        # Ensure faces and landmarks are lists (not None)
        if faces is None:
            faces = []
        if landmarks_list is None:
            landmarks_list = []
        
        return True
    
    def _draw_guidelines_state(self, frame, faces, landmarks_list):
        """Draw UI for guidelines state"""
        frame = self.liveness_detector.draw_guidelines(frame)
        
        # Check compliance if single face detected
        if len(faces) == 1 and len(landmarks_list) == 1:
            compliance_status = self.face_detector.get_compliance_status(landmarks_list[0], frame)
            self.liveness_detector.update_compliance_status(compliance_status)
            frame = self.liveness_detector.draw_compliance_status(frame, compliance_status)
        
        return frame
    
    def _draw_photo_review_state(self, frame):
        """Draw UI for photo review state"""
        return self.liveness_detector.draw_photo_review(frame)
    
    def _draw_common_ui_elements(self, frame, faces, landmarks_list, state):
        """Draw common UI elements for all states"""
        # Draw face boundaries and landmarks
        for i, face in enumerate(faces):
            if i < len(landmarks_list):
                frame = self.face_detector.draw_face_boundary(frame, face) or frame
                frame = self.face_detector.draw_landmarks(frame, landmarks_list[i]) or frame
        
        # Draw liveness detection UI
        frame = self.liveness_detector.draw_face_guide(frame) or frame
        frame = self.liveness_detector.draw_progress(frame) or frame
        
        # Draw head pose debug for specific states
        if (state.name in ['LOOK_LEFT', 'LOOK_RIGHT'] and 
            landmarks_list and len(landmarks_list) > 0):
            frame = self.liveness_detector.draw_head_pose_debug(frame, landmarks_list[0]) or frame
        
        # Draw countdown
        frame = self.draw_countdown(frame)
        
        return frame
    
    def _create_fallback_frame(self):
        """Create a fallback frame when camera fails"""
        fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(fallback_frame, "Camera Error - Press Q to quit", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return fallback_frame
    
    def _draw_state_indicator_and_controls(self, frame, state):
        """Draw state indicator and control instructions"""
        h, w = frame.shape[:2]
        ui_scale = self.get_ui_scale_factor(frame)
        
        # Draw state indicator circle
        state_color = self.liveness_detector.get_state_color()
        circle_radius = int(20 * ui_scale)
        circle_x = w - int(50 * ui_scale)
        circle_y = int(50 * ui_scale)
        
        cv2.circle(frame, (circle_x, circle_y), circle_radius, state_color, -1)
        cv2.circle(frame, (circle_x, circle_y), circle_radius, (255, 255, 255), max(1, int(2 * ui_scale)))
        
        # Draw controls text for fullscreen mode
        if w > 1400:
            controls_text = "Q: Quit  |  R: Restart  |  F: Exit Fullscreen"
            cv2.putText(frame, controls_text, (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def run(self, camera_index=0):
        """Main camera loop with optimized error handling and processing"""
        self.is_running = True
        
        # Performance and safety settings
        consecutive_errors = 0
        max_consecutive_errors = 10
        frame_count = 0
        process_every_n_frames = 3
        face_detection_interval = 0.1  # Detect faces every 100ms (10 FPS for detection)
        last_face_detection_time = 0
        
        try:
            self._initialize_camera_safely(camera_index)
            window_name = self._setup_display_window()
            is_fullscreen = True
            
            print("Camera window created in fullscreen mode")
            print("Press 'F' to exit fullscreen, 'ESC' to exit fullscreen, 'Q' to quit")
            
            while self.is_running:
                try:
                    # Process single frame iteration
                    frame_processed = self._process_single_frame(
                        frame_count, time.time(), last_face_detection_time, 
                        face_detection_interval, consecutive_errors, max_consecutive_errors
                    )
                    
                    if not frame_processed:
                        consecutive_errors += 1
                        if consecutive_errors >= max_consecutive_errors:
                            break
                        continue
                    
                    # Reset error counter on successful processing
                    consecutive_errors = 0
                    frame_count += 1
                    
                    # Update timing for face detection
                    current_time = time.time()
                    if (current_time - last_face_detection_time) >= face_detection_interval:
                        last_face_detection_time = current_time
                    
                    # Handle key events
                    if self._handle_key_events(window_name, is_fullscreen):
                        break
                    
                    # Small delay to maintain reasonable frame rate
                    time.sleep(0.01)  # 10ms delay for ~100 FPS display
                    
                except Exception as e:
                    consecutive_errors = self._handle_processing_error(e, consecutive_errors, max_consecutive_errors)
                    if consecutive_errors >= max_consecutive_errors:
                        break
                    
        except Exception as e:
            print(f"Critical error in camera loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup_camera()

    def _setup_display_window(self):
        """Setup and configure the display window"""
        window_name = 'Face Recognition with Liveness Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        return window_name

    def _process_single_frame(self, frame_count, current_time, last_face_detection_time, 
                             face_detection_interval, consecutive_errors, max_consecutive_errors):
        """Process a single frame with all necessary operations"""
        try:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                return False
            
            # Mirror frame
            frame = cv2.flip(frame, 1)
            
            # Get cached or fresh detection results
            faces, landmarks_list, state = self._get_detection_results(
                frame, current_time, last_face_detection_time, face_detection_interval
            )
            
            # Handle liveness state transitions
            self._handle_liveness_transitions(state, faces, landmarks_list, frame)
            
            # Handle photo capture countdown and capture
            self._handle_photo_capture(frame, landmarks_list)
            
            # Draw UI elements
            frame = self.draw_ui_elements(frame, faces, landmarks_list, state)
            
            # Handle photo review auto-proceed
            self._handle_photo_review_auto_proceed()
            
            # Display frame
            cv2.imshow('Face Recognition with Liveness Detection', frame)
            
            return True
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return False

    def _handle_key_events(self, window_name, is_fullscreen):
        """Handle all keyboard input events"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("Q pressed - quitting...")
            return True
        elif key == ord('f') or key == ord('F'):
            is_fullscreen = not is_fullscreen
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 
                               cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)
            print(f"Fullscreen: {'ON' if is_fullscreen else 'OFF'}")
        elif key == 27:  # ESC key
            if is_fullscreen:
                is_fullscreen = False
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            print("Fullscreen: OFF (ESC pressed)")
        elif key == ord('r') or key == ord('R'):
            self._handle_restart_command()
        elif key == ord('m') or key == ord('M'):
            self._handle_manual_advance()
        elif key == ord('s') or key == ord('S'):
            self._handle_emergency_skip()
        elif key == ord('a') or key == ord('A'):
            self._handle_manual_approval()
        
        return False

    def _handle_restart_command(self):
        """Handle restart command (R key)"""
        print("R pressed - restarting...")
        self.liveness_detector.reset()
        self.capture_countdown = 0
        self.photo_captured = False
        self.captured_frame = None

    def _handle_manual_advance(self):
        """Handle manual advance command (M key)"""
        self.liveness_detector.manual_advance()
        print("Manual advance triggered")

    def _handle_emergency_skip(self):
        """Handle emergency skip command (S key)"""
        if hasattr(self.liveness_detector, 'state'):
            print(f"Current state: {self.liveness_detector.state.name}")
            if self.liveness_detector.state.name in ['LOOK_LEFT', 'LOOK_RIGHT', 'BLINK']:
                self.liveness_detector.state = self.liveness_detector.LivenessState.COMPLETED
                print("EMERGENCY: Skipped to COMPLETED state")

    def _handle_manual_approval(self):
        """Handle manual approval command (A key)"""
        if (hasattr(self.liveness_detector, 'state') and 
            self.liveness_detector.state.name == 'PHOTO_REVIEW'):
            print("✅ Manual approval - Proceeding to completion...")
            # Save the photo after manual approval
            saved_path = self.save_captured_photo()
            if saved_path:
                self.liveness_detector.mark_capture_complete()
            else:
                print("❌ Failed to save photo - cannot proceed")
        else:
            print("Manual approval only available during photo review")

    def _initialize_camera_safely(self, camera_index):
        """Safely initialize camera with retry logic and performance optimization"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.initialize_camera(camera_index)
                print(f"Camera initialized successfully on attempt {attempt + 1}")
                
                # Optimize camera buffer settings for real-time performance
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer delay
                self.cap.set(cv2.CAP_PROP_FPS, 30)  # Ensure consistent FPS
                
                return
            except Exception as e:
                print(f"Camera initialization attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                else:
                    raise Exception(f"Camera initialization failed after {max_retries} attempts")

    def _get_detection_results(self, frame, current_time, last_face_detection_time, face_detection_interval):
        """Get face detection results with intelligent caching and performance optimization"""
        # Only perform face detection at specified intervals for performance
        if (current_time - last_face_detection_time) >= face_detection_interval:
            try:
                # Perform face detection with optimized parameters
                faces, landmarks_list = self.face_detector.detect_faces_optimized(frame)
                state = self.liveness_detector.get_current_state()
                
                # Cache results for next frame
                self._cached_detection_results = (faces, landmarks_list, state)
                self._last_detection_time = current_time
                
                return faces, landmarks_list, state
            except Exception as e:
                print(f"Face detection error: {e}")
                # Return cached results if available, otherwise empty results
                if hasattr(self, '_cached_detection_results'):
                    return self._cached_detection_results
                return [], [], self.liveness_detector.get_current_state()
        else:
            # Return cached results for performance
            if hasattr(self, '_cached_detection_results'):
                return self._cached_detection_results
            return [], [], self.liveness_detector.get_current_state()
    
    def _are_cached_results_valid(self, faces, landmarks_list, state):
        """Check if cached detection results are valid"""
        return (faces is not None and landmarks_list is not None and 
                state is not None and hasattr(state, 'name'))
    
    def _extract_landmarks(self, gray, faces):
        """Extract landmarks from detected faces"""
        landmarks_list = []
        for face in faces:
            try:
                landmarks = self.face_detector.get_landmarks(gray, face)
                landmarks_list.append(landmarks)
            except Exception as e:
                print(f"Landmark extraction error: {e}")
        return landmarks_list
    
    def _handle_liveness_transitions(self, state, faces, landmarks_list, frame):
        """Handle liveness detection state transitions efficiently"""
        if not state or not hasattr(state, 'name'):
            return
            
        state_name = state.name
        
        # Check compliance when waiting for face
        if (state_name == 'WAITING_FOR_FACE' and faces and landmarks_list and 
            len(faces) == 1 and len(landmarks_list) == 1):
            try:
                compliance_status = self.face_detector.get_compliance_status(landmarks_list[0], frame)
                if self.liveness_detector.should_return_to_guidelines(compliance_status):
                    self._reset_capture_state()
            except Exception as e:
                print(f"Compliance check error: {e}")
        
        # Handle photo capture initiation
        if (state_name == 'COMPLETED' and self.capture_countdown == 0 and not self.photo_captured):
            print("🎯 Starting photo capture countdown...")
            self.start_capture_countdown()
        elif (state_name == 'COMPLETED' and self.photo_captured):
            print("📸 Photo captured! Moving to completion state...")
            self.liveness_detector.mark_capture_complete()
    
    def _reset_capture_state(self):
        """Reset photo capture state"""
        self.liveness_detector.reset()
        self.capture_countdown = 0
        self.photo_captured = False
        self.captured_frame = None
    
    def _handle_photo_capture(self, frame, landmarks_list):
        """Handle photo capture initiation and countdown"""
        if self.capture_countdown > 0:
            self.update_countdown()
            if self.capture_countdown == 0:
                print("Countdown finished! Initiating photo capture...")
                if landmarks_list and len(landmarks_list) == 1:
                    # Capture photo
                    captured_photo = self.capture_photo(frame, landmarks_list[0])
                    print("✅ Photo captured successfully! Starting quality check...")
                    
                    # Quality check
                    try:
                        quality_issues = self.face_detector.analyze_captured_photo_quality(captured_photo, landmarks_list[0])
                        
                        if quality_issues:
                            print(f"⚠️  Quality Issues Detected: {len(quality_issues)} issues")
                            self.liveness_detector.start_photo_review(quality_issues)
                        else:
                            print("✅ PHOTO QUALITY APPROVED - PROCESS COMPLETE")
                            # Save the photo after quality approval
                            saved_path = self.save_captured_photo()
                            if saved_path:
                                self.liveness_detector.mark_capture_complete()
                            else:
                                print("❌ Failed to save photo - cannot proceed")
                    except Exception as e:
                        print(f"Error in quality check: {e}")
                        # Continue without quality check if it fails
                        self.liveness_detector.mark_capture_complete()
                else:
                    print("No valid landmarks for photo capture.")
        elif self.photo_captured:
            # If countdown is 0 and photo is captured, save it
            if self.captured_frame is not None:
                self.save_captured_photo()
                self.reset_photo_capture() # Reset for next photo
    
    def _handle_photo_review_auto_proceed(self):
        """Automatically proceed to the next state if photo review is completed."""
        if (self.liveness_detector.state.name == 'PHOTO_REVIEW' and 
            not self.liveness_detector.has_photo_quality_issues()):
            # Quality check completed successfully, auto-proceed after delay
            if not hasattr(self, '_photo_review_start_time'):
                self._photo_review_start_time = time.time()
            
            if time.time() - self._photo_review_start_time > 3:  # 3 seconds delay
                print("✅ Quality check completed - Auto-proceeding to completion...")
                # Save the photo after auto-approval
                saved_path = self.save_captured_photo()
                if saved_path:
                    self.liveness_detector.mark_capture_complete()
                else:
                    print("❌ Failed to save photo - cannot proceed")
                # Reset timer
                self._photo_review_start_time = None
    

    

    

    
    def _create_fallback_frame(self):
        """Create a fallback frame when camera fails"""
        fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(fallback_frame, "Camera Error - Press Q to quit", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return fallback_frame
    
    def _handle_camera_error(self, consecutive_errors, max_consecutive_errors):
        """Handle camera errors with intelligent recovery"""
        consecutive_errors += 1
        print(f"Camera error #{consecutive_errors}/{max_consecutive_errors}")
        
        if consecutive_errors >= max_consecutive_errors:
            print("Maximum camera errors reached - stopping camera")
        else:
            # Try to recover camera
            try:
                if self.cap and self.cap.isOpened():
                    self.cap.release()
                time.sleep(0.1)  # Brief pause before retry
                self.initialize_camera(0)  # Retry with default camera
                consecutive_errors = 0  # Reset on successful recovery
                print("Camera recovered successfully")
            except Exception as e:
                print(f"Camera recovery failed: {e}")
        
        return consecutive_errors

    def _handle_processing_error(self, error, consecutive_errors, max_consecutive_errors):
        """Handle processing errors with intelligent recovery"""
        consecutive_errors += 1
        print(f"Processing error #{consecutive_errors}/{max_consecutive_errors}: {error}")
        
        if consecutive_errors >= max_consecutive_errors:
            print("Maximum processing errors reached - stopping camera")
        else:
            # Try to recover processing state
            try:
                # Reset critical components
                if hasattr(self, 'liveness_detector'):
                    self.liveness_detector.reset()
                self.capture_countdown = 0
                self.photo_captured = False
                consecutive_errors = 0
                print("Processing state recovered successfully")
            except Exception as e:
                print(f"Processing recovery failed: {e}")
        
        return consecutive_errors

    def _cleanup_camera(self):
        """Enhanced camera cleanup with memory management"""
        try:
            if hasattr(self, 'cap') and self.cap:
                if self.cap.isOpened():
                    self.cap.release()
                    print("Camera released successfully")
                self.cap = None
            
            # Close all OpenCV windows
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Ensure windows are closed
            
            print("Camera cleanup completed successfully")
            
        except Exception as e:
            print(f"Error during camera cleanup: {e}")
        finally:
            # Ensure camera is released
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
                self.cap = None
            
    def stop(self):
        print("Stopping camera interface...")
        self.is_running = False
        # Force cleanup if stop is called externally
        self.release_camera()
        cv2.destroyAllWindows()