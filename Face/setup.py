#for setup 
import os
import sys
import subprocess
import urllib.request
import bz2
import shutil

def install_requirements():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True
    except subprocess.CalledProcessError as e:
        return False

def download_dlib_landmarks():
    landmarks_file = "shape_predictor_68_face_landmarks.dat"
    
    if os.path.exists(landmarks_file):
        return True
    
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    compressed_file = "shape_predictor_68_face_landmarks.dat.bz2"
    
    try:
        urllib.request.urlretrieve(url, compressed_file)

        with bz2.BZ2File(compressed_file, 'rb') as source:
            with open(landmarks_file, 'wb') as target:
                shutil.copyfileobj(source, target)
        
        os.remove(compressed_file)
        
        return True
        
    except Exception as e:
        return False

def check_cmake():
    try:
        subprocess.check_output(['cmake', '--version'])

        return True
    except (subprocess.CalledProcessError, FileNotFoundError):

        return False

def main():
    success = True
    
    if not check_cmake():
        success = False
    
    if success and not install_requirements():
        success = False
    
    if success and not download_dlib_landmarks():
        success = False
    
    if success:
        print("Setup completed successfully!")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()