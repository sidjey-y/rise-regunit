#!/usr/bin/env python3

import numpy as np
from deepface import DeepFace

def test_deepface_represent():
    """Test DeepFace.represent() function"""
    try:
        print("Testing DeepFace.represent()...")
        
        # Test with a simple call
        embedding = DeepFace.represent(
            img_path="raw_pic/face_embeddings.jpg",
            model_name="VGG-Face",
            enforce_detection=True
        )
        
        if embedding:
            embedding_array = np.array(embedding)
            print(f"✓ Successfully extracted embedding (shape: {embedding_array.shape})")
            print(f"✓ Embedding type: {type(embedding_array)}")
            print(f"✓ First 5 values: {embedding_array[:5]}")
            return True
        else:
            print("✗ No embedding returned")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    test_deepface_represent() 