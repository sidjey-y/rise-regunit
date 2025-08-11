#!/usr/bin/env python3

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

class LivenessDetectorDebug:
    def __init__(self, face_detector):
        self.face_detector = face_detector
        self.guidelines_shown_time = 0
        self.reset()
        
        # ADJUSTED Configuration parameters for better detection
        self.DIRECTION_THRESHOLD = 8.0  # Reduced from 15.0 to 8.0 degrees
        self.DIRECTION_FRAMES_REQUIRED = 5  # Reduced from 10 to 5 frames
        self.BLINK_FRAMES_REQUIRED = 3
        self.MAX_TIME_PER_STEP = 15  # Increased from 10 to 15 seconds
        self.FACE_STABILITY_FRAMES = 5
        
        self.direction_frame_count = 0
        self.blink_frame_count = 0
        self.face_stable_count = 0
        self.last_ear_values = deque(maxlen=10)
        
        self.compliance_stable_count = 0
        self.COMPLIANCE_FRAMES_REQUIRED = 15
        self.last_compliance_status = None
        self.compliance_passed = False
        
        # Debug tracking
        self.photo_quality_issues = []
        self.photo_review_start_time = 0
        self.debug_log = []
        
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
        self.compliance_passed = False
        
        # Clear debug log
        self.debug_log.clear()
        
    def log_debug(self, message):
        """Log debug information with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        debug_msg = f"[{timestamp}] {message}"
        self.debug_log.append(debug_msg)
        print(debug_msg)
        
        # Keep only last 50 debug messages
        if len(self.debug_log) > 50:
            self.debug_log.pop(0)
    
    def get_current_instruction(self):
        instructions = {
            LivenessState.SHOW_GUIDELINES: "Follow the guidelines and fix any issues shown. Process will continue automatically when compliant.",
            LivenessState.WAITING_FOR_FACE: "Position your face within the oval",
            LivenessState.BLINK: "Stay in the oval and BLINK your eyes",
            LivenessState.LOOK_LEFT: "Stay in the oval and SLOWLY turn your head LEFT",
            LivenessState.LOOK_RIGHT: "Stay in the oval and SLOWLY turn your head RIGHT", 
            LivenessState.COMPLETED: "Stay still - Capturing...",
            LivenessState.FAILED: "Keep your face in the oval. Starting over...",
            LivenessState.CAPTURE_COMPLETE: "Facial Capture Complete!",
            LivenessState.PHOTO_REVIEW: "Photo quality check in progress..."
        }
        return instructions.get(self.state, "")
    
    def is_face_centered(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        
        # Face center
        face_center_x = np.mean(landmarks[:, 0])
        face_center_y = np.mean(landmarks[:, 1])
        
        # If face is in center region (middle 60% of frame)
        center_x_min, center_x_max = w * 0.2, w * 0.8
        center_y_min, center_y_max = h * 0.2, h * 0.8
        
        return (center_x_min <= face_center_x <= center_x_max and 
                center_y_min <= face_center_y <= center_y_max)
    
    def is_face_appropriate_size(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        
        # Face dimensions
        face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
        face_height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
        
        # Face should be between 20% and 80% of frame dimensions
        min_width = w * 0.2
        max_width = w * 0.8
        min_height = h * 0.2
        max_height = h * 0.8
        
        return (min_width <= face_width <= max_width and 
                min_height <= face_height <= max_height)
    
    def detect_smooth_blink(self, ear):
        """Detect smooth blink with improved logic"""
        self.last_ear_values.append(ear)
        
        if len(self.last_ear_values) < 3:
            return False
        
        # Check for blink pattern: high -> low -> high
        recent_values = list(self.last_ear_values)[-3:]
        
        # Blink detected if: high -> low -> high pattern
        if (recent_values[0] > 0.25 and 
            recent_values[1] < 0.25 and 
            recent_values[2] > 0.25):
            return True
        
        return False
    
    def update(self, frame, faces, landmarks_list):
        """Main update method with enhanced debugging"""
        current_time = time.time()
        
        # Check for timeout
        if current_time - self.step_start_time > self.MAX_TIME_PER_STEP:
            self.log_debug(f"Step timeout after {self.MAX_TIME_PER_STEP}s, resetting...")
            self.reset()
            return self.state
        
        # State machine logic
        if self.state == LivenessState.SHOW_GUIDELINES:
            # Wait for guidelines to be shown
            if current_time - self.guidelines_shown_time > 2:
                self.state = LivenessState.WAITING_FOR_FACE
                self.step_start_time = current_time
                self.log_debug("Moving to WAITING_FOR_FACE state")
                
        elif self.state == LivenessState.WAITING_FOR_FACE:
            if len(faces) == 1 and len(landmarks_list) == 1:
                landmarks = landmarks_list[0]
                
                if (self.is_face_centered(landmarks, frame.shape) and 
                    self.is_face_appropriate_size(landmarks, frame.shape)):
                    self.face_stable_count += 1
                    if self.face_stable_count >= self.FACE_STABILITY_FRAMES:
                        self.completed_steps['face_detected'] = True
                        self.state = LivenessState.BLINK
                        self.step_start_time = current_time
                        self.log_debug("Face detected and stable, moving to BLINK state")
                else:
                    self.face_stable_count = 0
                    
        elif self.state == LivenessState.BLINK:
            if len(landmarks_list) > 0:
                landmarks = landmarks_list[0]
                is_blinking, ear = self.face_detector.is_blinking(landmarks)
                
                # Enhanced blink detection
                if self.detect_smooth_blink(ear):
                    self.completed_steps['blinked'] = True
                    self.state = LivenessState.LOOK_LEFT
                    self.step_start_time = current_time
                    self.direction_frame_count = 0
                    self.log_debug("Blink detected, moving to LOOK_LEFT state")
                    
        elif self.state == LivenessState.LOOK_LEFT:
            if len(landmarks_list) > 0:
                landmarks = landmarks_list[0]
                direction = self.face_detector.get_face_direction(landmarks)
                pitch, yaw, roll = self.face_detector.get_head_pose(landmarks, frame.shape)
                
                # Enhanced left detection with debugging
                self.log_debug(f"LOOK_LEFT: Direction={direction}, Yaw={yaw:.1f}°, Threshold={-self.DIRECTION_THRESHOLD}°")
                
                if direction == "left" or (yaw is not None and yaw < -self.DIRECTION_THRESHOLD):
                    self.direction_frame_count += 1
                    self.log_debug(f"Left turn detected: {self.direction_frame_count}/{self.DIRECTION_FRAMES_REQUIRED} frames")
                    
                    if self.direction_frame_count >= self.DIRECTION_FRAMES_REQUIRED:
                        self.completed_steps['looked_left'] = True
                        self.state = LivenessState.LOOK_RIGHT
                        self.step_start_time = current_time
                        self.direction_frame_count = 0
                        self.log_debug("Left turn completed, moving to LOOK_RIGHT state")
                else:
                    if self.direction_frame_count > 0:
                        self.log_debug(f"Left turn reset: face returned to center")
                    self.direction_frame_count = 0
                    
        elif self.state == LivenessState.LOOK_RIGHT:
            if len(landmarks_list) > 0:
                landmarks = landmarks_list[0]
                direction = self.face_detector.get_face_direction(landmarks)
                pitch, yaw, roll = self.face_detector.get_head_pose(landmarks, frame.shape)
                
                # Enhanced right detection with debugging
                self.log_debug(f"LOOK_RIGHT: Direction={direction}, Yaw={yaw:.1f}°, Threshold={self.DIRECTION_THRESHOLD}°")
                
                if direction == "right" or (yaw is not None and yaw > self.DIRECTION_THRESHOLD):
                    self.direction_frame_count += 1
                    self.log_debug(f"Right turn detected: {self.direction_frame_count}/{self.DIRECTION_FRAMES_REQUIRED} frames")
                    
                    if self.direction_frame_count >= self.DIRECTION_FRAMES_REQUIRED:
                        self.completed_steps['looked_right'] = True
                        self.state = LivenessState.COMPLETED
                        self.step_start_time = current_time
                        self.log_debug("Right turn completed, moving to COMPLETED state")
                else:
                    if self.direction_frame_count > 0:
                        self.log_debug(f"Right turn reset: face returned to center")
                    self.direction_frame_count = 0
        
        return self.state
    
    def mark_capture_complete(self):
        self.state = LivenessState.CAPTURE_COMPLETE
        self.log_debug("Capture marked as complete")
    
    def start_photo_review(self, photo_quality_issues):
        self.state = LivenessState.PHOTO_REVIEW
        self.photo_quality_issues = photo_quality_issues
        self.photo_review_start_time = time.time()
        self.log_debug(f"Photo review started with {len(photo_quality_issues)} issues")
    
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
            self.log_debug("Manually proceeding from guidelines")
    
    def update_compliance_status(self, compliance_status):
        if self.state != LivenessState.SHOW_GUIDELINES:
            return
        
        if compliance_status and compliance_status['compliant']:
            if self.last_compliance_status and self.last_compliance_status['compliant']:
                self.compliance_stable_count += 1
            else:
                self.compliance_stable_count = 1
            
            if self.compliance_stable_count >= self.COMPLIANCE_FRAMES_REQUIRED:
                self.state = LivenessState.WAITING_FOR_FACE
                self.step_start_time = time.time()
                self.face_stable_count = 0
                self.last_ear_values.clear()
                self.compliance_stable_count = 0
                self.compliance_passed = True
                self.log_debug("Compliance requirements met, auto-progressing")
        else:
            self.compliance_stable_count = 0
        
        self.last_compliance_status = compliance_status
    
    def should_return_to_guidelines(self, compliance_status):
        if not compliance_status:
            return False
        
        if self.compliance_passed:
            return compliance_status.get('eyeglasses_detected', False)
        
        return False
    
    def get_debug_info(self):
        """Get debug information for display"""
        return {
            'state': self.state.name,
            'direction_frame_count': self.direction_frame_count,
            'completed_steps': self.completed_steps,
            'threshold': self.DIRECTION_THRESHOLD,
            'frames_required': self.DIRECTION_FRAMES_REQUIRED,
            'debug_log': self.debug_log[-5:]  # Last 5 debug messages
        }

