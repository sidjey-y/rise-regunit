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
        
        # Configuration parameters
        self.DIRECTION_THRESHOLD = 15  # degrees for head pose
        self.DIRECTION_FRAMES_REQUIRED = 10  # consecutive frames required
        self.BLINK_FRAMES_REQUIRED = 3  # consecutive frames for blink
        self.MAX_TIME_PER_STEP = 10  # seconds
        self.FACE_STABILITY_FRAMES = 5  # frames to confirm face presence
        
        self.direction_frame_count = 0
        self.blink_frame_count = 0
        self.face_stable_count = 0
        self.last_ear_values = deque(maxlen=10)
        
        self.compliance_stable_count = 0
        self.COMPLIANCE_FRAMES_REQUIRED = 15  # 0.5 seconds at 30fps (reduced from 30)
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
            LivenessState.LOOK_LEFT: "Stay in the oval and SLOWLY turn your head LEFT",
            LivenessState.LOOK_RIGHT: "Stay in the oval and SLOWLYturn your head RIGHT", 
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
        
        #if face is in center region (middle 60% of frame)
        center_x_min, center_x_max = w * 0.2, w * 0.8
        center_y_min, center_y_max = h * 0.2, h * 0.8
        
        return (center_x_min <= face_center_x <= center_x_max and 
                center_y_min <= face_center_y <= center_y_max)
    
    #face is not too close and not too farr
    def is_face_appropriate_size(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        
        #face dimensions
        face_width = np.max(landmarks[:, 0]) - np.min(landmarks[:, 0])
        face_height = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
        
        # face should occupy 20-60% of frame width ata
        min_width, max_width = w * 0.2, w * 0.6
        min_height, max_height = h * 0.2, h * 0.6
        
        return (min_width <= face_width <= max_width and 
                min_height <= face_height <= max_height)
    
    def detect_smooth_blink(self, ear):
        self.last_ear_values.append(ear)
        
        if len(self.last_ear_values) < 6:
            return False
            
        # look for blink pattern: normal -> closing -> closed -> opening -> normal
        values = list(self.last_ear_values)
        
        # significant drop and recovery in EAR
        max_ear = max(values[-6:-2])  
        min_ear = min(values[-4:-1]) 
        current_ear = values[-1]      
        
        # Blink pattern: high -> low -> recovering
        if (max_ear > 0.25 and min_ear < 0.2 and 
            current_ear > min_ear + 0.05 and 
            max_ear - min_ear > 0.1):
            return True
            
        return False
    
    def update(self, frame, faces, landmarks_list):
        current_time = time.time()
        
        # for timeout
        if current_time - self.step_start_time > self.MAX_TIME_PER_STEP:
            self.state = LivenessState.FAILED
            return self.state
        
        # no face detected
        if len(faces) == 0:
            if self.state != LivenessState.WAITING_FOR_FACE:
                self.face_stable_count = 0
            return self.state
        
        # multiple faces detected
        if len(faces) > 1:
            return self.state  # stay in current state, don't progress
        
        # single face detected
        landmarks = landmarks_list[0]
        
        # face quality
        face_is_valid = (self.is_face_centered(landmarks, frame.shape) and 
                        self.is_face_appropriate_size(landmarks, frame.shape))
        
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
            is_blinking, ear = self.face_detector.is_blinking(landmarks)
            
            #  blink detection
            if self.detect_smooth_blink(ear):
                self.completed_steps['blinked'] = True
                self.state = LivenessState.LOOK_LEFT
                self.step_start_time = current_time
                self.direction_frame_count = 0
                
        elif self.state == LivenessState.LOOK_LEFT:
            direction = self.face_detector.get_face_direction(landmarks)
            pitch, yaw, roll = self.face_detector.get_head_pose(landmarks, frame.shape)
            
            # if looking left (negative yaw)
            if direction == "left" or (yaw is not None and yaw < -self.DIRECTION_THRESHOLD):
                self.direction_frame_count += 1
                if self.direction_frame_count >= self.DIRECTION_FRAMES_REQUIRED:
                    self.completed_steps['looked_left'] = True
                    self.state = LivenessState.LOOK_RIGHT
                    self.step_start_time = current_time
                    self.direction_frame_count = 0
            else:
                self.direction_frame_count = 0
                
        elif self.state == LivenessState.LOOK_RIGHT:
            direction = self.face_detector.get_face_direction(landmarks)
            pitch, yaw, roll = self.face_detector.get_head_pose(landmarks, frame.shape)
            
            # if looking right (positive yaw)
            if direction == "right" or (yaw is not None and yaw > self.DIRECTION_THRESHOLD):
                self.direction_frame_count += 1
                if self.direction_frame_count >= self.DIRECTION_FRAMES_REQUIRED:
                    self.completed_steps['looked_right'] = True
                    self.state = LivenessState.COMPLETED
                    self.step_start_time = current_time
            else:
                self.direction_frame_count = 0
        
        return self.state
    
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
    
    def draw_progress(self, frame):
        h, w = frame.shape[:2]
        
        instruction = self.get_current_instruction()
        
        completed_count = sum(self.completed_steps.values())
        total_steps = len(self.completed_steps) - 1  
        current_step = completed_count if completed_count < total_steps else total_steps
        
        step_text = f"Step {current_step}/3"
        if self.state == LivenessState.COMPLETED:
            step_text = "COMPLETE"
        elif self.state == LivenessState.CAPTURE_COMPLETE:
            step_text = "CAPTURE COMPLETE"
        elif self.state == LivenessState.PHOTO_REVIEW:
            step_text = "QUALITY CHECK"
        elif self.state == LivenessState.FAILED:
            step_text = "FAILED"
        elif self.state == LivenessState.WAITING_FOR_FACE and completed_count > 0:
            step_text = "STAY IN OVAL!"
        
        #  color based on state
        if self.state == LivenessState.WAITING_FOR_FACE and completed_count > 0:
            text_color = (0, 0, 255)  # Red warning
        elif self.state == LivenessState.COMPLETED:
            text_color = (0, 255, 0)  # Yellow
        elif self.state == LivenessState.CAPTURE_COMPLETE:
            text_color = (0, 255, 0)  # Yellow
        elif self.state == LivenessState.FAILED:
            text_color = (0, 0, 255)  # Red
        else:
            text_color = (255, 255, 255)  # White
        
        cv2.putText(frame, step_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
        
        return frame
    
    def draw_guidelines(self, frame):
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
            "<< BASIC CHECKS MUST BE GREEN >>",
        ]
        
        # starting position for left side (guidelines)
        font_scale = 0.7
        thickness = 2
        line_height = 30
        start_y = 50
        start_x = 50
        
        # draw each line
        for i, line in enumerate(guidelines):
            y = start_y + (i * line_height)
            
            if line == "PHOTO REQUIREMENTS":
                font_scale_title = 1.1
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
        h, w = frame.shape[:2]
        
        # right side area 
        right_start_x = w // 2 + 50
        start_y = 100
        
        cv2.putText(frame, "REAL-TIME CHECK", (right_start_x, start_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        y_offset = start_y + 50
        
        if compliance_status is None:
            # No face detected
            cv2.putText(frame, "No face detected", (right_start_x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, "Please position yourself", (right_start_x, y_offset + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "in front of the camera", (right_start_x, y_offset + 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            face_issues = compliance_status['face_coverage_issues']
            status_items = [
                ("Face clear:", "✓ Clear" if len(face_issues) == 0 else f"✗ {len(face_issues)} issue(s) detected",
                 (0, 255, 0) if len(face_issues) == 0 else (0, 0, 255))
            ]
            
            for item_label, item_status, color in status_items:
                cv2.putText(frame, item_label, (right_start_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, item_status, (right_start_x + 120, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_offset += 40
            
            #specific issues
            if compliance_status['issues']:
                y_offset += 20
                cv2.putText(frame, "ISSUES TO FIX:", (right_start_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                y_offset += 30
                
                for issue in compliance_status['issues'][:3]: 
                    cv2.putText(frame, f"• {issue}", (right_start_x, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
                    y_offset += 25
            
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
                    overall_text = f"✓ COMPLIANT - {remaining_seconds:.1f}s to proceed"
                    overall_color = (0, 255, 0)
                    
                    # progress bar
                    bar_width = 200
                    bar_height = 10
                    bar_x = right_start_x
                    bar_y = y_offset + 40
                    
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)


                    progress_width = int(bar_width * progress_ratio)
                    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), (0, 255, 0), -1)
                    
                    cv2.putText(frame, "Auto-progression", (right_start_x, y_offset + 65), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                overall_text = "✗ FIX ALL ISSUES ABOVE"
                overall_color = (0, 0, 255)
            
            cv2.putText(frame, overall_text, (right_start_x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, overall_color, 2)
        
        return frame

    def draw_photo_review(self, frame):
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        
        # Title
        title = "Photo Quality Review"
        font_scale = 1.2
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)[0]
        title_x = (w - text_size[0]) // 2
        cv2.putText(frame, title, (title_x, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (0, 255, 255), 3)
        
        y_offset = 150
            # No issues found
        cv2.putText(frame, "✓ PHOTO QUALITY APPROVED", (50, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        y_offset += 80
            
        cv2.putText(frame, "Photo meets quality requirements", (50, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        y_offset += 60
            
        cv2.putText(frame, "Proceeding to completion...", (50, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame
    
    def draw_face_guide(self, frame):
        h, w = frame.shape[:2]
        
        # oval guide throughout the entire process
        center_x, center_y = w // 2, h // 2 - 50
        axes = (int(w * 0.15), int(h * 0.25))
        
        # change oval color based on state
        if self.state == LivenessState.COMPLETED:
            oval_color = (0, 255, 0)  # green for success
        elif self.state == LivenessState.CAPTURE_COMPLETE:
            oval_color = (0, 255, 0)  # green for capture complete
        elif self.state == LivenessState.FAILED:
            oval_color = (0, 0, 255)  # red for failure
        else:
            oval_color = (0, 255, 255)  # yellow during liveness checks
            
        cv2.ellipse(frame, (center_x, center_y), axes, 0, 0, 360, oval_color, 3)
        
        # draw instruction text
        instruction = self.get_current_instruction()
        
        if self.state == LivenessState.COMPLETED:
            font_scale = 1.5
            color = (0, 255, 255)  #yellow
        elif self.state == LivenessState.CAPTURE_COMPLETE:
            font_scale = 1.5
            color = (0, 255, 0)  
        elif self.state == LivenessState.FAILED:
            font_scale = 1.2
            color = (0, 0, 255)  
        else:
            font_scale = 1.2
            color = (255, 255, 255) 
        
        text_size = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h - 80
        
        padding = 20
        cv2.rectangle(frame, (text_x - padding, text_y - 40), 
                     (text_x + text_size[0] + padding, text_y + 15), (0, 0, 0), -1)
        cv2.putText(frame, instruction, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 3)
        
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