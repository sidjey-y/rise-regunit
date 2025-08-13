#!/usr/bin/env python3
"""
Performance Monitoring Script for Face Recognition System
Use this to monitor and optimize performance.
"""

import time
import psutil
import cv2
import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_detector import FaceDetector
from config_manager import ConfigManager

class PerformanceMonitor:
    def __init__(self):
        self.frame_times = []
        self.processing_times = []
        self.start_time = time.time()
        self.frame_count = 0
        
    def start_frame(self):
        self.frame_start_time = time.time()
        
    def end_frame(self):
        frame_time = time.time() - self.frame_start_time
        self.frame_times.append(frame_time)
        self.frame_count += 1
        
        # Keep only last 100 frames for rolling average
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)
    
    def start_processing(self):
        self.processing_start_time = time.time()
        
    def end_processing(self):
        processing_time = time.time() - self.processing_start_time
        self.processing_times.append(processing_time)
        
        # Keep only last 100 processing times
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
    
    def get_stats(self):
        if not self.frame_times:
            return {}
            
        avg_frame_time = np.mean(self.frame_times)
        avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
        
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        uptime = time.time() - self.start_time
        
        return {
            'avg_fps': avg_fps,
            'avg_frame_time': avg_frame_time * 1000,  # Convert to ms
            'avg_processing_time': avg_processing_time * 1000,  # Convert to ms
            'total_frames': self.frame_count,
            'uptime': uptime,
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent
        }

def main():
    print("Performance Monitoring for Face Recognition System")
    print("=" * 60)
    print("This script will monitor performance metrics in real-time.")
    print("Press 'Q' to quit, 'R' to reset stats.")
    print("=" * 60)
    
    try:
        # Initialize components
        config_manager = ConfigManager("config.yaml")
        face_detector = FaceDetector(config_manager)
        
        if not face_detector.initialize():
            print("Failed to initialize face detector")
            return
        
        # Get camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Failed to open camera")
            return
        
        # Set camera properties for performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize performance monitor
        monitor = PerformanceMonitor()
        
        # Performance test parameters
        test_duration = 30  # seconds
        test_start_time = time.time()
        
        print(f"Starting performance test for {test_duration} seconds...")
        print("Turn your head slowly to test head pose detection performance.")
        
        while True:
            monitor.start_frame()
            
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Mirror the frame
            frame = cv2.flip(frame, 1)
            
            # Monitor face detection performance
            monitor.start_processing()
            faces, gray = face_detector.detect_faces(frame)
            monitor.end_processing()
            
            if len(faces) > 0:
                landmarks = face_detector.get_landmarks(gray, faces[0])
                
                # Monitor head pose performance
                monitor.start_processing()
                pitch, yaw, roll = face_detector.get_head_pose(landmarks, frame.shape)
                monitor.end_processing()
                
                # Draw landmarks
                frame = face_detector.draw_landmarks(frame, landmarks)
                
                # Display head pose values
                if yaw is not None:
                    cv2.putText(frame, f"Yaw: {yaw:.1f}°", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Pitch: {pitch:.1f}°", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Roll: {roll:.1f}°", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Get performance stats
            stats = monitor.get_stats()
            
            # Display performance metrics
            y_offset = 150
            cv2.putText(frame, f"FPS: {stats['avg_fps']:.1f}", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_offset += 30
            cv2.putText(frame, f"Frame Time: {stats['avg_frame_time']:.1f}ms", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_offset += 30
            cv2.putText(frame, f"Processing: {stats['avg_processing_time']:.1f}ms", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_offset += 30
            cv2.putText(frame, f"CPU: {stats['cpu_percent']:.1f}%", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_offset += 30
            cv2.putText(frame, f"Memory: {stats['memory_percent']:.1f}%", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_offset += 30
            cv2.putText(frame, f"Uptime: {stats['uptime']:.1f}s", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Check if test duration is complete
            elapsed_time = time.time() - test_start_time
            if elapsed_time >= test_duration:
                print(f"\nPerformance test completed after {elapsed_time:.1f} seconds")
                print(f"Final Stats:")
                print(f"  Average FPS: {stats['avg_fps']:.1f}")
                print(f"  Average Frame Time: {stats['avg_frame_time']:.1f}ms")
                print(f"  Average Processing Time: {stats['avg_processing_time']:.1f}ms")
                print(f"  Total Frames Processed: {stats['total_frames']}")
                print(f"  CPU Usage: {stats['cpu_percent']:.1f}%")
                print(f"  Memory Usage: {stats['memory_percent']:.1f}%")
                break
            
            # Display instructions
            cv2.putText(frame, "Performance Test Running...", (10, frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Time Remaining: {test_duration - elapsed_time:.1f}s", (10, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Performance Monitor', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('r') or key == ord('R'):
                # Reset monitor
                monitor = PerformanceMonitor()
                test_start_time = time.time()
                print("Performance monitor reset")
            
            monitor.end_frame()
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()



