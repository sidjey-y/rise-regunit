import cv2
import time
import numpy as np
from enum import Enum
from collections import deque

class LivenessState(Enum):
    SHOW_GUIDELINES = 0
    WAITING_FOR_FACE = 1
    BLINK = 2
    LOOK_LEFT = 3
    LOOK_RIGHT = 4
    COMPLETED = 5
    FAILED = 6
    CAPTURE_COMPLETE = 7
    PHOTO_REVIEW = 8

class LivenessDetector:
    def __init__(self, face_detector):
        self.face_detector = face_detector
        self.guidelines_shown_time = 0
        self.reset()
        
        # Manual override for testing (set to True to skip head turns)
        self.MANUAL_OVERRIDE = False
        
        # Configuration parameters - get from config if available
        if hasattr(face_detector, 'config_manager') and face_detector.config_manager:
            liveness_config = face_detector.config_manager.get('liveness', {})
            self.DIRECTION_THRESHOLD = liveness_config.get('head_movement_threshold', 12.0)  # Increased from 6.0 - less sensitive
            self.DIRECTION_FRAMES_REQUIRED = liveness_config.get('direction_frames_required', 10)  # Increased from 6 - more stable
            self.BLINK_FRAMES_REQUIRED = liveness_config.get('blink_frames_required', 3)
            # self.MAX_TIME_PER_STEP = liveness_config.get('max_time_per_step', 30)  # DISABLED - No timeout
            self.FACE_STABILITY_FRAMES = liveness_config.get('face_stability_frames', 5)
            self.AUTO_RESTART_ON_FAILURE = liveness_config.get('auto_restart_on_failure', True)
            self.RESTART_DELAY = liveness_config.get('restart_delay', 3.0)
            self.DEBUG_MODE = liveness_config.get('debug', {}).get('show_head_pose_values', True)
        else:
            self.DIRECTION_THRESHOLD = 12.0  # Increased from 6.0 - less sensitive, requires more head turn
            self.DIRECTION_FRAMES_REQUIRED = 10  # Increased from 6 - more frames required for stability
            self.BLINK_FRAMES_REQUIRED = 3
            # self.MAX_TIME_PER_STEP = 30  # DISABLED - No timeout for better user experience
            self.FACE_STABILITY_FRAMES = 5
            self.AUTO_RESTART_ON_FAILURE = True
            self.RESTART_DELAY = 3.0
            self.DEBUG_MODE = True
        
        self.direction_frame_count = 0
        self.blink_frame_count = 0
        self.face_stable_count = 0
        self.last_ear_values = deque(maxlen=10)
        
        self.compliance_stable_count = 0
        self.COMPLIANCE_FRAMES_REQUIRED = 45  # Increased from 15 - now requires 1.5 seconds at 30fps (was 0.5 seconds)
        self.last_compliance_status = None
        self.compliance_passed = False  # track if user already passed compliance
        
        # reset photo quality review
        self.photo_quality_issues = []
        self.photo_review_start_time = 0
        
    def reset(self):
        self.state = LivenessState.SHOW_GUIDELINES
        self.start_time = time.time()
        self.step_start_time = time.time()
        self.direction_frame_count = 0
        self.blink_frame_count = 0
        self.face_stable_count = 0
        if hasattr(self, 'last_ear_values'):
            self.last_ear_values.clear()
        else:
            self.last_ear_values = deque(maxlen=10)
        self.completed_steps = {
            'face_detected': False,
            'blinked': False,
            'looked_left': False,
            'looked_right': False
        }
        self.guidelines_shown_time = time.time()
        
        self.compliance_stable_count = 0
        self.last_compliance_status = None
        self.compliance_passed = False  # if user already passed compliance
        
    def get_current_instruction(self):
        instructions = {
            LivenessState.SHOW_GUIDELINES: "Follow the guidelines and fix any issues shown. Process will continue automatically when compliant.",
            LivenessState.WAITING_FOR_FACE: "Position your face within the oval",
            LivenessState.BLINK: "Stay in the oval and BLINK your eyes",
            LivenessState.LOOK_LEFT: "Stay in the oval and turn your head LEFT (turn more)",
            LivenessState.LOOK_RIGHT: "Stay in the oval and turn your head RIGHT (turn more)", 
            LivenessState.COMPLETED: "Stay still - Capturing...",
            LivenessState.FAILED: "Keep your face in the oval. Starting over...",
            LivenessState.CAPTURE_COMPLETE: "Facial Capture Complete!",
            LivenessState.PHOTO_REVIEW: "Photo quality check in progress..."
        }
        return instructions.get(self.state, "")
    
    def is_face_centered(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        
        #face center
        face_center_x = np.mean(landmarks[:, 0])
        face_center_y = np.mean(landmarks[:, 1])
        
        #More lenient center region (middle 70% of frame) to match larger oval
        center_x_min, center_x_max = w * 0.15, w * 0.85  # Increased from 0.2-0.8 to 0.15-0.85
        center_y_min, center_y_max = h * 0.15, h * 0.85  # Increased from 0.2-0.8 to 0.15-0.85
        
        return (center_x_min <= face_center_x <= center_x_max and 
                center_y_min <= face_center_y <= center_y_max)
    
    #face is not too close and not too farr
    def is_face_appropriate_size(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        
        #face dimensions
        face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
        face_height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
        
        # More lenient face size requirements to match larger oval (20% x 30%)
        # Allow face to be 15-75% of frame width and 15-75% of frame height
        min_width, max_width = w * 0.15, w * 0.75   # Increased from 0.2-0.6 to 0.15-0.75
        min_height, max_height = h * 0.15, h * 0.75  # Increased from 0.2-0.6 to 0.15-0.75
        
        return (min_width <= face_width <= max_width and 
                min_height <= face_height <= max_height)
    
    def detect_smooth_blink(self, ear):
        """Detect blink with improved sensitivity and reliability"""
        try:
            # Validate input
            if ear is None or ear <= 0 or np.isnan(ear) or np.isinf(ear):
                return False
            
            self.last_ear_values.append(ear)
            
            # Need at least 3 frames for detection
            if len(self.last_ear_values) < 3:
                return False
                
            # Get recent values
            values = list(self.last_ear_values)
            
            # Validate all values in the window
            if any(v is None or v <= 0 or np.isnan(v) or np.isinf(v) for v in values[-3:]):
                return False
            
            # Use a 3-frame window for more responsive detection
            current_ear = values[-1]      # Current frame
            previous_ear = values[-2]     # Previous frame
            older_ear = values[-3]        # 3 frames ago
            
            # Validate calculated values
            if current_ear <= 0 or previous_ear <= 0 or older_ear <= 0:
                return False
            
            # MORE SENSITIVE thresholds for better blink detection
            OPEN_THRESHOLD = 0.20   # Reduced from 0.21 - Eyes are considered open above this
            CLOSED_THRESHOLD = 0.18 # Reduced from 0.19 - Eyes are considered closed below this
            MIN_CHANGE = 0.02       # Reduced from 0.03 - Smaller change needed to detect blink
            
            # Check if this looks like a blink pattern
            eyes_were_open = older_ear > OPEN_THRESHOLD
            eyes_closed = previous_ear < CLOSED_THRESHOLD
            eyes_opened_again = current_ear > OPEN_THRESHOLD
            
            # Calculate the change magnitude
            change_magnitude = max(older_ear, current_ear) - previous_ear
            
            # Debug logging - ALWAYS show this for troubleshooting
            print(f"🔍 Blink Debug - Open: {eyes_were_open}, Closed: {eyes_closed}, Opened: {eyes_opened_again}")
            print(f"🔍 Blink Debug - Values: {older_ear:.3f} -> {previous_ear:.3f} -> {current_ear:.3f}")
            print(f"🔍 Blink Debug - Change: {change_magnitude:.3f} (min: {MIN_CHANGE})")
            print(f"🔍 Blink Debug - Thresholds: Open>{OPEN_THRESHOLD}, Closed<{CLOSED_THRESHOLD}")
            
            # Blink detected if all conditions are met
            if (eyes_were_open and eyes_closed and eyes_opened_again and 
                change_magnitude > MIN_CHANGE):
                
                print(f"✅ BLINK DETECTED! Pattern: {older_ear:.3f} -> {previous_ear:.3f} -> {current_ear:.3f}")
                # Clear the buffer after successful detection
                self.last_ear_values.clear()
                return True
                
            return False
            
        except Exception as e:
            print(f"Error in detect_smooth_blink: {e}")
            return False
    
    def _detect_simple_blink(self):
        try:
            if len(self.last_ear_values) < 5:
                return False
            
            values = list(self.last_ear_values)[-5:]
            
            max_ear = max(values)
            min_ear = min(values)
            current_ear = values[-1]
            
            if (max_ear > 0.20 and  # Eyes were open
                min_ear < 0.18 and   # Eyes closed at some point
                current_ear > 0.20 and  # Eyes are open now
                (max_ear - min_ear) > 0.04):  # Significant change
                
                print(f"✅ Simple blink detected: max={max_ear:.3f}, min={min_ear:.3f}, current={current_ear:.3f}")
                self.last_ear_values.clear()
                return True
            
            return False
            
        except Exception as e:
            print(f"Error in _detect_simple_blink: {e}")
            return False
    
    def update(self, frame, faces, landmarks_list):
        current_time = time.time()
        
        # No timeout check - users can take their time
        # Timeout functionality completely disabled for better user experience
        
        # no face detected
        if len(faces) == 0:
            if self.state != LivenessState.WAITING_FOR_FACE:
                self.face_stable_count = 0
            return self.state
        
        # multiple faces detected
        if len(faces) > 1:
            return self.state  # stay in current state, don't progress
        
        # single face detected - validate landmarks
        if len(landmarks_list) == 0:
            print("No landmarks available for face")
            return self.state
            
        landmarks = landmarks_list[0]
        
        # Validate landmarks
        if landmarks is None or len(landmarks) < 48:
            print(f"Invalid landmarks: {landmarks is None}, length: {len(landmarks) if landmarks is not None else 0}")
            return self.state
        
        # Check for invalid landmark coordinates
        try:
            if np.any(np.isnan(landmarks)) or np.any(np.isinf(landmarks)):
                print("Landmarks contain NaN or infinite values")
                return self.state
        except Exception as e:
            print(f"Error validating landmarks: {e}")
            return self.state
        
        # face quality with debug information
        try:
            is_centered = self.is_face_centered(landmarks, frame.shape)
            is_appropriate_size = self.is_face_appropriate_size(landmarks, frame.shape)
            face_is_valid = is_centered and is_appropriate_size
            
            # Debug information when face becomes invalid
            if not face_is_valid:
                h, w = frame.shape[:2]
                face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
                face_height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
                face_center_x = np.mean(landmarks[:, 0])
                face_center_y = np.mean(landmarks[:, 1])
                
                print(f"🔍 Face validation failed:")
                print(f"   • Centered: {is_centered} (face center: {face_center_x:.0f}, {face_center_y:.0f})")
                print(f"   • Size OK: {is_appropriate_size} (face size: {face_width:.0f}x{face_height:.0f} = {face_width/w*100:.1f}%x{face_height/h*100:.1f}%)")
                print(f"   • Valid ranges: width 15-75%, height 15-75%")
                
        except Exception as e:
            print(f"Error checking face validity: {e}")
            face_is_valid = False
        
        if not face_is_valid:
            self.face_stable_count = 0
            if self.state != LivenessState.WAITING_FOR_FACE:

                # reset if face becomes invalid during liveness check
                self.state = LivenessState.WAITING_FOR_FACE
                self.step_start_time = current_time

                #reset all progress
                self.completed_steps = {
                    'face_detected': False,
                    'blinked': False,
                    'looked_left': False,
                    'looked_right': False
                }
            return self.state
        
        
        if self.state == LivenessState.SHOW_GUIDELINES:
            
            # automatic progression based on compliance
            return self.state

        elif self.state == LivenessState.WAITING_FOR_FACE:
            self.face_stable_count += 1
            if self.face_stable_count >= self.FACE_STABILITY_FRAMES:
                self.completed_steps['face_detected'] = True
                self.state = LivenessState.BLINK
                self.step_start_time = current_time
                self.blink_frame_count = 0
                self.last_ear_values.clear()
                
        elif self.state == LivenessState.BLINK:
            try:
                # Get blink detection with error handling
                is_blinking, ear = self.face_detector.is_blinking(landmarks)
                
                # Enhanced debug logging for troubleshooting
                if self.DEBUG_MODE and self.blink_frame_count % 5 == 0:  # Log every 5 frames
                    print(f"BLINK - EAR: {ear:.3f}, Buffer size: {len(self.last_ear_values)}")
                    if len(self.last_ear_values) > 0:
                        # Convert deque to list for slicing
                        ear_list = list(self.last_ear_values)
                        recent_values = ear_list[-4:] if len(ear_list) >= 4 else ear_list
                        print(f"BLINK - Recent EAR values: {[f'{v:.3f}' for v in recent_values]}")
                
                # Validate EAR value
                if ear is None or ear <= 0 or np.isnan(ear) or np.isinf(ear):
                    print(f"Invalid EAR value: {ear}, skipping blink detection")
                    return self.state
                
                # Add EAR to tracking with validation
                if 0 < ear < 1.0:  # Valid EAR range
                    self.last_ear_values.append(ear)
                    self.blink_frame_count += 1
                else:
                    print(f"EAR out of valid range: {ear}")
                    return self.state
                
                # Try primary blink detection method
                if self.detect_smooth_blink(ear):
                    self.completed_steps['blinked'] = True
                    self.state = LivenessState.LOOK_LEFT
                    self.step_start_time = current_time
                    self.direction_frame_count = 0
                    print("✅ Blink detected! Moving to LOOK_LEFT")
                    return self.state
                
                # Fallback: Simple blink detection if buffer is full
                if len(self.last_ear_values) >= 5:
                    if self._detect_simple_blink():
                        self.completed_steps['blinked'] = True
                        self.state = LivenessState.LOOK_LEFT
                        self.step_start_time = current_time
                        self.direction_frame_count = 0
                        print("✅ Simple blink detected! Moving to LOOK_LEFT")
                        return self.state
                
                # No timeout protection for blink - users can take their time
                # Blink timeout functionality disabled for better user experience
                    
            except Exception as e:
                print(f"Error in BLINK state: {e}")
                import traceback
                traceback.print_exc()
                # Don't crash, just stay in current state
                return self.state
        
        elif self.state == LivenessState.LOOK_LEFT:
            # Get head pose (yaw angle) - this is more reliable than face direction
            pitch, yaw, roll = self.face_detector.get_head_pose(landmarks, frame.shape)
            
            # Enhanced debug logging for troubleshooting
            if self.DEBUG_MODE and self.direction_frame_count % 5 == 0:  # Log every 5 frames
                if yaw is not None:
                    print(f"LOOK_LEFT - Yaw: {yaw:.2f}°, Threshold: -{self.DIRECTION_THRESHOLD}°, Progress: {self.direction_frame_count}/{self.DIRECTION_FRAMES_REQUIRED}")
                else:
                    print(f"LOOK_LEFT - Yaw: None (head pose detection failed)")
            
            # Check if looking left (negative yaw means left turn) - less sensitive now
            if yaw is not None and yaw < -self.DIRECTION_THRESHOLD:
                self.direction_frame_count += 1
                if self.direction_frame_count % 5 == 0:  # Log every 5 frames
                    print(f"LOOK_LEFT - Detected left turn: {yaw:.2f}° (frame {self.direction_frame_count}/{self.DIRECTION_FRAMES_REQUIRED})")
                if self.direction_frame_count >= self.DIRECTION_FRAMES_REQUIRED:
                    self.completed_steps['looked_left'] = True
                    self.state = LivenessState.LOOK_RIGHT
                    self.step_start_time = current_time
                    self.direction_frame_count = 0
                    print("LOOK_LEFT - Completed! Moving to LOOK_RIGHT")
            else:
                if self.direction_frame_count > 0 and self.direction_frame_count % 5 == 0:
                    if yaw is not None:
                        print(f"LOOK_LEFT - Reset counter (yaw: {yaw:.2f}° not < -{self.DIRECTION_THRESHOLD}°)")
                    else:
                        print(f"LOOK_LEFT - Reset counter (head pose detection failed)")
                self.direction_frame_count = 0
                
        elif self.state == LivenessState.LOOK_RIGHT:
            # Get head pose (yaw angle) - this is more reliable than face direction
            pitch, yaw, roll = self.face_detector.get_head_pose(landmarks, frame.shape)
            
            # Enhanced debug logging for troubleshooting
            if self.DEBUG_MODE and self.direction_frame_count % 5 == 0:  # Log every 5 frames
                if yaw is not None:
                    print(f"LOOK_RIGHT - Yaw: {yaw:.2f}°, Threshold: +{self.DIRECTION_THRESHOLD}°, Progress: {self.direction_frame_count}/{self.DIRECTION_FRAMES_REQUIRED}")
                else:
                    print(f"LOOK_RIGHT - Yaw: None (head pose detection failed)")
            
            # Check if looking right (positive yaw means right turn) - less sensitive now
            if yaw is not None and yaw > self.DIRECTION_THRESHOLD:
                self.direction_frame_count += 1
                if self.direction_frame_count % 5 == 0:  # Log every 5 frames
                    print(f"LOOK_RIGHT - Detected right turn: {yaw:.2f}° (frame {self.direction_frame_count}/{self.DIRECTION_FRAMES_REQUIRED})")
                if self.direction_frame_count >= self.DIRECTION_FRAMES_REQUIRED:
                    self.completed_steps['looked_right'] = True
                    self.state = LivenessState.COMPLETED
                    self.step_start_time = current_time
                    print("LOOK_RIGHT - Completed! Moving to COMPLETED")
            else:
                if self.direction_frame_count > 0 and self.direction_frame_count % 5 == 0:
                    if yaw is not None:
                        print(f"LOOK_RIGHT - Reset counter (yaw: {yaw:.2f}° not > +{self.DIRECTION_THRESHOLD}°)")
                    else:
                        print(f"LOOK_RIGHT - Reset counter (head pose detection failed)")
                self.direction_frame_count = 0
        
        # Check if liveness test should be restarted due to failure
        if self.state == LivenessState.FAILED:
            if self.AUTO_RESTART_ON_FAILURE:
                print(f"Liveness test failed. Auto-restarting in {self.RESTART_DELAY} seconds...")
                time.sleep(self.RESTART_DELAY)
                self.reset()
                print("Liveness test restarted automatically")
                return self.state  # Return the new state after reset
            else:
                print("Liveness test failed. Manual restart required.")
                # Force auto-restart anyway to prevent getting stuck
                print("Forcing auto-restart to prevent system lockup...")
                self.reset()
                print("Liveness test restarted automatically")
                return self.state
        
        # No timeout protection - users can take their time with head turns
        # Timeout functionality disabled for better user experience
        
        return self.state
    
    def handle_failed_state(self):
        """Handle failed state and automatically restart liveness test"""
        if self.AUTO_RESTART_ON_FAILURE:
            print(f"Liveness test failed. Auto-restarting in {self.RESTART_DELAY} seconds...")
            time.sleep(self.RESTART_DELAY)
            self.reset()
            print("Liveness test restarted automatically")
        else:
            print("Liveness test failed. Manual restart required.")
    
    def mark_capture_complete(self):
        self.state = LivenessState.CAPTURE_COMPLETE
    
    def start_photo_review(self, photo_quality_issues):
        self.state = LivenessState.PHOTO_REVIEW
        self.photo_quality_issues = photo_quality_issues
        self.photo_review_start_time = time.time()
    
    def get_photo_quality_issues(self):
        return self.photo_quality_issues
    
    def has_photo_quality_issues(self):
        return len(self.photo_quality_issues) > 0
    
    def proceed_from_guidelines(self):
        if self.state == LivenessState.SHOW_GUIDELINES:
            self.state = LivenessState.WAITING_FOR_FACE
            self.step_start_time = time.time()
            self.face_stable_count = 0
            self.last_ear_values.clear()
    
    def update_compliance_status(self, compliance_status):
        if self.state != LivenessState.SHOW_GUIDELINES:
            return
        
        if compliance_status and compliance_status['compliant']:
            # Compliance is good
            if self.last_compliance_status and self.last_compliance_status['compliant']:
                # Previously compliant, increment stable count
                self.compliance_stable_count += 1
            else:
                # Just became compliant, reset counter
                self.compliance_stable_count = 1
            
            # auto-progress if stable compliance achieved
            if self.compliance_stable_count >= self.COMPLIANCE_FRAMES_REQUIRED:
                self.state = LivenessState.WAITING_FOR_FACE
                self.step_start_time = time.time()
                self.face_stable_count = 0
                self.last_ear_values.clear()
                self.compliance_stable_count = 0
                self.compliance_passed = True  #passed
        else:
            #not compliant, reset counter
            self.compliance_stable_count = 0
        
        self.last_compliance_status = compliance_status
    
    def should_return_to_guidelines(self, compliance_status):
        if not compliance_status:
            return False
        
        # If compliance was already passed
        if self.compliance_passed:
            # return for extremely critical violations (ex.b like putting glasses back on)
            return compliance_status['eyeglasses_detected']
        
        # only return to guidelines for critical violations during early liveness states
        # don't interrupt if user is in middle of liveness sequence
        if self.state in [LivenessState.BLINK, LivenessState.LOOK_LEFT, LivenessState.LOOK_RIGHT]:
            # only for severe violations
            critical_violations = [
                compliance_status['eyeglasses_detected'],
                # return for very severe face coverage (multiple issues)
                len(compliance_status['face_coverage_issues']) > 1
            ]
            return any(critical_violations)
        

        critical_violations = [
            compliance_status['eyeglasses_detected'],
            len(compliance_status['face_coverage_issues']) > 0
        ]
        
        return any(critical_violations)
    
    def should_restart_liveness(self):
        """Check if liveness test should be restarted due to failure"""
        # Restart if in failed state and auto-restart is enabled
        if self.state == LivenessState.FAILED and self.AUTO_RESTART_ON_FAILURE:
            return True
        
        # No timeout-based restart - users can take their time
        # Timeout functionality completely disabled
        
        return False
    
    def manual_advance(self):
        """Manually advance to next liveness step (for testing/debugging)"""
        current_time = time.time()
        
        if self.state == LivenessState.LOOK_LEFT:
            print("MANUAL OVERRIDE: Completing LOOK_LEFT")
            self.completed_steps['looked_left'] = True
            self.state = LivenessState.LOOK_RIGHT
            self.step_start_time = current_time
            self.direction_frame_count = 0
        elif self.state == LivenessState.LOOK_RIGHT:
            print("MANUAL OVERRIDE: Completing LOOK_RIGHT")
            self.completed_steps['looked_right'] = True
            self.state = LivenessState.COMPLETED
            self.step_start_time = current_time
        elif self.state == LivenessState.BLINK:
            print("MANUAL OVERRIDE: Completing BLINK")
            self.completed_steps['blinked'] = True
            self.state = LivenessState.LOOK_LEFT
            self.step_start_time = current_time
            self.direction_frame_count = 0
        else:
            print(f"Manual advance not available in state: {self.state.name}")
    
    def draw_progress(self, frame):
        # Safety check: ensure frame is not None
        if frame is None:
            print("Warning: draw_progress called with None frame")
            fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(fallback_frame, "Camera Error - Press Q to quit", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return fallback_frame
            
        h, w = frame.shape[:2]
        
        instruction = self.get_current_instruction()
        
        completed_count = sum(self.completed_steps.values())
        total_steps = len(self.completed_steps) - 1  
        current_step = completed_count if completed_count < total_steps else total_steps
        
        step_text = f"Step {current_step}/3"
        if self.state == LivenessState.COMPLETED:
            step_text = "CAPTURING..."
        elif self.state == LivenessState.CAPTURE_COMPLETE:
            step_text = "COMPLETE ✓"
        elif self.state == LivenessState.PHOTO_REVIEW:
            step_text = "REVIEWING..."
        elif self.state == LivenessState.FAILED:
            step_text = "FAILED ✗"
        elif self.state == LivenessState.WAITING_FOR_FACE and completed_count > 0:
            step_text = "STAY IN POSITION!"
        
        #  color based on state
        if self.state == LivenessState.WAITING_FOR_FACE and completed_count > 0:
            text_color = (0, 100, 255)  # Orange warning
            bg_color = (0, 0, 0)
        elif self.state == LivenessState.COMPLETED:
            text_color = (0, 255, 255)  # Cyan for capturing
            bg_color = (0, 0, 0)
        elif self.state == LivenessState.CAPTURE_COMPLETE:
            text_color = (0, 255, 0)  # Green for success
            bg_color = (0, 0, 0)
        elif self.state == LivenessState.FAILED:
            text_color = (0, 0, 255)  # Red for failure
            bg_color = (0, 0, 0)
        else:
            text_color = (255, 255, 255)  # White for normal
            bg_color = (0, 0, 0)
        
        # Better font scaling based on resolution
        base_font_scale = 1.0
        if w > 1400:  # High resolution
            font_scale = base_font_scale * 1.2
        elif w > 1000:  # Medium resolution
            font_scale = base_font_scale
        else:  # Low resolution
            font_scale = base_font_scale * 0.8
        
        thickness = max(2, int(2 if w > 1400 else 2))
        
        # Modern progress display with background
        text_size = cv2.getTextSize(step_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        
        # Position in top-left with modern styling
        padding = 15
        bg_x1, bg_y1 = 10, 10
        bg_x2, bg_y2 = text_size[0] + 40, 60
        
        # Semi-transparent modern background
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add subtle border
        border_color = (100, 100, 100) if self.state not in [LivenessState.FAILED, LivenessState.CAPTURE_COMPLETE] else text_color
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), border_color, 2)
        
        cv2.putText(frame, step_text, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, text_color, thickness)
        
        return frame
    
    def draw_guidelines(self, frame):
        # Safety check: ensure frame is not None
        if frame is None:
            print("Warning: draw_guidelines called with None frame")
            fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(fallback_frame, "Camera Error - Press Q to quit", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return fallback_frame
            
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        
        # Guidelines text - practical requirements
        guidelines = [
            "PHOTO REQUIREMENTS",
            "",
            "BASIC REQUIREMENTS:",
            "- Remove eyeglasses if wearing",
            "- Keep forehead visible (no heavy bangs)",
            "- No objects covering face",
            "- Look directly at camera lens",
            "",
            "POSE GUIDELINES:",
            "- Head straight (no tilting/rotation)",
            "- Face camera frontally",
            "- Neutral expression, eyes open",
            "- Mouth closed (slight smile OK, no teeth)",
            "",
            "IMPORTANT: Hold steady position for 1.5 seconds",
            "when all checks are green to proceed automatically",
            "",
            "<< BASIC CHECKS MUST BE GREEN >>",
        ]
        
        # starting position for left side (guidelines)
        # Better font scaling based on resolution
        if w > 1400:  # High resolution
            font_scale = 0.8
            font_scale_title = 1.3
            thickness = 2
            line_height = 35
        elif w > 1000:  # Medium resolution
            font_scale = 0.7
            font_scale_title = 1.1
            thickness = 2
            line_height = 30
        else:  # Low resolution
            font_scale = 0.6
            font_scale_title = 0.9
            thickness = 2
            line_height = 25
            
        start_y = 50
        start_x = 50
        
        # draw each line
        for i, line in enumerate(guidelines):
            y = start_y + (i * line_height)
            
            if line == "PHOTO REQUIREMENTS":
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale_title, 3)[0]
                x = (w//2 - text_size[0]) // 2 
                cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale_title, (255, 255, 0), 3)
            elif line.startswith("BASIC") or line.startswith("POSE"):
                cv2.putText(frame, line, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, (0, 255, 255), thickness)
            elif line.startswith("•"):
                cv2.putText(frame, line, (start_x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, (255, 255, 255), thickness)
            elif "BASIC CHECKS MUST BE GREEN" in line:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                x = (w//2 - text_size[0]) // 2
                cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, (0, 165, 255), thickness)
            elif "IMPORTANT:" in line:
                cv2.putText(frame, line, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, (255, 165, 0), thickness)  # Orange for important info
            elif "when all checks are green" in line:
                cv2.putText(frame, line, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, (255, 165, 0), thickness)  # Orange for important info
            elif "Process continues automatically" in line:
                text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                x = (w//2 - text_size[0]) // 2
                cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, (0, 255, 0), thickness)
            elif line == "":
                continue
            else:
                cv2.putText(frame, line, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, (255, 255, 255), thickness)
        
        cv2.line(frame, (w//2, 0), (w//2, h), (100, 100, 100), 2)
        
        return frame
    
    #real time comp;iance status
    def draw_compliance_status(self, frame, compliance_status=None):
        # Safety check: ensure frame is not None
        if frame is None:
            print("Warning: draw_compliance_status called with None frame")
            fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(fallback_frame, "Camera Error - Press Q to quit", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return fallback_frame
            
        h, w = frame.shape[:2]
        
        # right side area with better scaling
        right_start_x = w // 2 + 50
        start_y = 100
        
        # Adaptive font scaling
        if w > 1400:  # High resolution
            title_font_scale = 1.2
            main_font_scale = 0.9
            small_font_scale = 0.7
        elif w > 1000:  # Medium resolution
            title_font_scale = 1.0
            main_font_scale = 0.8
            small_font_scale = 0.6
        else:  # Low resolution
            title_font_scale = 0.8
            main_font_scale = 0.7
            small_font_scale = 0.5
        
        cv2.putText(frame, "REAL-TIME CHECK", (right_start_x, start_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, (0, 255, 255), 2)
        
        y_offset = start_y + 50
        
        if compliance_status is None:
            # No face detected
            cv2.putText(frame, "No face detected", (right_start_x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, main_font_scale, (0, 0, 255), 2)
            cv2.putText(frame, "Please position yourself", (right_start_x, y_offset + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, main_font_scale, (255, 255, 255), 2)
            cv2.putText(frame, "in front of the camera", (right_start_x, y_offset + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, main_font_scale, (255, 255, 255), 2)
        else:
            face_issues = compliance_status['face_coverage_issues']
            status_items = [
                ("Face clear:", "✓ Clear" if len(face_issues) == 0 else f"✗ {len(face_issues)} issue(s) detected",
                 (0, 255, 0) if len(face_issues) == 0 else (0, 0, 255))
            ]
            
            for item_label, item_status, color in status_items:
                cv2.putText(frame, item_label, (right_start_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, small_font_scale, (255, 255, 255), 2)
                cv2.putText(frame, item_status, (right_start_x + int(120 * (main_font_scale/0.8)), y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, small_font_scale, color, 2)
                y_offset += int(40 * (main_font_scale/0.8))
            
            #specific issues
            if compliance_status['issues']:
                y_offset += 20
                cv2.putText(frame, "ISSUES TO FIX:", (right_start_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, main_font_scale, (0, 0, 255), 2)
                y_offset += 30
                
                for issue in compliance_status['issues'][:3]: 
                    cv2.putText(frame, f"• {issue}", (right_start_x, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, small_font_scale, (255, 100, 100), 2)
                    y_offset += int(25 * (main_font_scale/0.8))
            
            #status with progression info
            y_offset += 30
            if compliance_status['compliant']:
                progress_ratio = self.compliance_stable_count / self.COMPLIANCE_FRAMES_REQUIRED
                remaining_frames = max(0, self.COMPLIANCE_FRAMES_REQUIRED - self.compliance_stable_count)
                remaining_seconds = remaining_frames / 30.0  # 30 fps
                
                if self.compliance_stable_count >= self.COMPLIANCE_FRAMES_REQUIRED:
                    overall_text = "✓ PROCEEDING TO LIVENESS..."
                    overall_color = (0, 255, 0)
                else:
                    overall_text = f"✓ COMPLIANT - {remaining_seconds:.1f}s remaining"
                    overall_color = (0, 255, 0)
                    
                    # progress bar
                    bar_width = 200
                    bar_height = 10
                    bar_x = right_start_x
                    bar_y = y_offset + 40
                    
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)


                    progress_width = int(bar_width * progress_ratio)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
                    
                    # Add progress percentage
                    progress_percentage = int(progress_ratio * 100)
                    cv2.putText(frame, f"{progress_percentage}%", (bar_x + bar_width + 10, bar_y + 8), 
                               cv2.FONT_HERSHEY_SIMPLEX, small_font_scale * 0.8, (255, 255, 255), 1)
                    
                    cv2.putText(frame, "Hold steady for auto-progression", (right_start_x, y_offset + 65), 
                               cv2.FONT_HERSHEY_SIMPLEX, small_font_scale, (255, 255, 255), 1)
            else:
                overall_text = "✗ FIX ALL ISSUES ABOVE"
                overall_color = (0, 0, 255)
            
            cv2.putText(frame, overall_text, (right_start_x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, main_font_scale, overall_color, 2)
        
        return frame

    def draw_photo_review(self, frame):
        # Safety check: ensure frame is not None
        if frame is None:
            print("Warning: draw_photo_review called with None frame")
            fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(fallback_frame, "Camera Error - Press Q to quit", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return fallback_frame
            
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        
        # Better scaling for photo review
        if w > 1400:  # High resolution
            title_font_scale = 1.4
            main_font_scale = 1.3
            sub_font_scale = 1.0
        elif w > 1000:  # Medium resolution
            title_font_scale = 1.2
            main_font_scale = 1.1
            sub_font_scale = 0.9
        else:  # Low resolution
            title_font_scale = 1.0
            main_font_scale = 0.9
            sub_font_scale = 0.8
        
        # Title
        title = "Photo Quality Review"
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_font_scale, 3)[0]
        title_x = (w - text_size[0]) // 2
        cv2.putText(frame, title, (title_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                   title_font_scale, (0, 255, 255), 3)
        
        y_offset = 150
        
        if len(self.photo_quality_issues) == 0:
            # No issues found
            cv2.putText(frame, "✓ PHOTO QUALITY APPROVED", (50, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, main_font_scale, (0, 255, 0), 3)
            y_offset += 80
                
            cv2.putText(frame, "Photo meets international standards:", (50, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, sub_font_scale, (255, 255, 255), 2)
            y_offset += 40
            
            # List the standards met
            standards = [
                "• ICAO 9303 (Passport photo requirements)",
                "• ISO/IEC 19794-5 (Facial image quality)",
                "• Government ID photo standards"
            ]
            
            for standard in standards:
                cv2.putText(frame, standard, (70, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, sub_font_scale * 0.8, (200, 255, 200), 2)
                y_offset += 30
                
            y_offset += 20
            cv2.putText(frame, "Proceeding to completion...", (50, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, sub_font_scale, (0, 255, 0), 2)
        else:
            # Quality issues detected
            cv2.putText(frame, "⚠ QUALITY ISSUES DETECTED", (50, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, main_font_scale, (0, 165, 255), 3)
            y_offset += 80
            
            cv2.putText(frame, f"Found {len(self.photo_quality_issues)} issue(s) to fix:", (50, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, sub_font_scale, (255, 255, 255), 2)
            y_offset += 50
            
            # Show first 5 issues
            for i, issue in enumerate(self.photo_quality_issues[:5]):
                cv2.putText(frame, f"{i+1}. {issue}", (70, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, sub_font_scale * 0.8, (255, 200, 100), 2)
                y_offset += 35
            
            if len(self.photo_quality_issues) > 5:
                cv2.putText(frame, f"... and {len(self.photo_quality_issues) - 5} more issues", 
                           (70, y_offset), cv2.FONT_HERSHEY_SIMPLEX, sub_font_scale * 0.8, (255, 200, 100), 2)
                y_offset += 35
            
            y_offset += 20
            cv2.putText(frame, "Press 'A' to approve anyway or 'R' to retake", (50, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, sub_font_scale, (255, 255, 0), 2)
        
        return frame
    
    def draw_face_guide(self, frame):
        # Safety check: ensure frame is not None
        if frame is None:
            print("Warning: draw_face_guide called with None frame")
            fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(fallback_frame, "Camera Error - Press Q to quit", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return fallback_frame
            
        h, w = frame.shape[:2]
        
        # Don't draw the oval during capture states to avoid UI clutter
        if self.state in [LivenessState.COMPLETED, LivenessState.CAPTURE_COMPLETE, LivenessState.PHOTO_REVIEW]:
            return frame
        
        # Optimized oval size - larger and more user-friendly
        center_x, center_y = w // 2, h // 2 - 40  # Moved up more to avoid bottom text
        axes = (int(w * 0.20), int(h * 0.30))  # Larger oval size - increased from 0.15, 0.22
        
        # change oval color based on state with better visual design
        if self.state == LivenessState.FAILED:
            oval_color = (0, 0, 255)  # red for failure
            thickness = 3
        elif self.state == LivenessState.WAITING_FOR_FACE:
            oval_color = (0, 255, 255)  # cyan for positioning
            thickness = 2
        else:
            oval_color = (255, 255, 0)  # yellow during liveness checks
            thickness = 3
            
        # Draw oval with subtle transparency effect
        overlay = frame.copy()
        cv2.ellipse(overlay, (center_x, center_y), axes, 0, 0, 360, oval_color, thickness)
        cv2.ellipse(overlay, (center_x, center_y), (axes[0]-10, axes[1]-10), 0, 0, 360, oval_color, 1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        # draw instruction text with better positioning and styling
        instruction = self.get_current_instruction()
        
        # Skip drawing instruction during capture states to reduce clutter
        if self.state in [LivenessState.COMPLETED, LivenessState.CAPTURE_COMPLETE, LivenessState.PHOTO_REVIEW]:
            return frame
        
        # Adjust font size based on screen resolution
        base_font_scale = 0.8
        if w > 1400:  # High resolution (1920x1080, etc.)
            font_scale = base_font_scale * 1.1
        elif w > 1000:  # Medium resolution
            font_scale = base_font_scale
        else:  # Low resolution
            font_scale = base_font_scale * 0.9
        
        if self.state == LivenessState.FAILED:
            color = (0, 0, 255)  # Red for failure
            font_scale *= 1.1  # Slightly larger for alerts
        else:
            color = (255, 255, 255)  # White for normal instructions
        
        # Position text at bottom with better spacing
        text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h - 80  # More space from bottom
        
        # Modern text background with rounded corners effect
        padding = 20
        bg_x1, bg_y1 = text_x - padding, text_y - 30
        bg_x2, bg_y2 = text_x + text_size[0] + padding, text_y + 15
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add subtle border
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (80, 80, 80), 2)
        
        cv2.putText(frame, instruction, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
        
        return frame
    
    def get_state_color(self):
        colors = {
            LivenessState.SHOW_GUIDELINES: (0, 255, 255),  # Yellow
            LivenessState.WAITING_FOR_FACE: (0, 255, 255),  # Yellow
            LivenessState.BLINK: (255, 0, 255),             # Magenta
            LivenessState.LOOK_LEFT: (255, 165, 0),         # Orange
            LivenessState.LOOK_RIGHT: (255, 165, 0),        # Orange
            LivenessState.COMPLETED: (0, 255, 0),           # Green
            LivenessState.FAILED: (0, 0, 255),              # Red
            LivenessState.CAPTURE_COMPLETE: (0, 255, 0),    # Green
            LivenessState.PHOTO_REVIEW: (0, 255, 255)       # Cyan
        }
        return colors.get(self.state, (255, 255, 255))

    def draw_head_pose_debug(self, frame, landmarks):
        """Draw real-time head pose values for debugging"""
        # Safety check: ensure frame is not None
        if frame is None:
            print("Warning: draw_head_pose_debug called with None frame")
            fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(fallback_frame, "Camera Error - Press Q to quit", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return fallback_frame
            
        try:
            pitch, yaw, roll = self.face_detector.get_head_pose(landmarks, frame.shape)
            
            if yaw is not None:
                # Display current head pose values
                cv2.putText(frame, f"Yaw: {yaw:.1f}°", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Pitch: {pitch:.1f}°", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Roll: {roll:.1f}°", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show threshold line with updated value
                cv2.putText(frame, f"Threshold: ±{self.DIRECTION_THRESHOLD:.0f}°", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Color code based on direction
                if yaw < -self.DIRECTION_THRESHOLD:
                    direction_color = (0, 0, 255)  # Red for left
                    direction_text = "LEFT"
                elif yaw > self.DIRECTION_THRESHOLD:
                    direction_color = (255, 0, 0)  # Blue for right
                    direction_text = "RIGHT"
                else:
                    direction_color = (0, 255, 0)  # Green for center
                    direction_text = "CENTER"
                
                cv2.putText(frame, f"Direction: {direction_text}", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, direction_color, 2)
                
        except Exception as e:
            cv2.putText(frame, f"Head pose error: {e}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def draw_blink_debug(self, frame, landmarks):
        """Draw real-time EAR values and blink status for debugging"""
        # Safety check: ensure frame is not None
        if frame is None:
            print("Warning: draw_blink_debug called with None frame")
            fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(fallback_frame, "Camera Error - Press Q to quit", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return fallback_frame
            
        try:
            # Get current EAR value
            is_blinking, ear = self.face_detector.is_blinking(landmarks)
            
            if ear is not None:
                # Display current EAR value
                cv2.putText(frame, f"EAR: {ear:.3f}", (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Color code based on blink state
                if is_blinking:
                    blink_color = (0, 0, 255)  # Red for blinking
                    blink_text = "BLINKING"
                else:
                    blink_color = (0, 255, 0)  # Green for eyes open
                    blink_text = "EYES OPEN"
                
                cv2.putText(frame, f"Status: {blink_text}", (10, 210), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, blink_color, 2)
                
                # Show buffer status
                buffer_size = len(self.last_ear_values)
                cv2.putText(frame, f"Buffer: {buffer_size}/10", (10, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show recent EAR values if available
                if buffer_size > 0:
                    recent_values = [f"{v:.3f}" for v in self.last_ear_values[-3:]]
                    cv2.putText(frame, f"Recent: {recent_values}", (10, 270), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Show thresholds
                cv2.putText(frame, f"Open: >0.21, Closed: <0.19", (10, 300), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                
        except Exception as e:
            cv2.putText(frame, f"Blink debug error: {e}", (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame