# Face Embedding Extraction System

This system extracts face embeddings from images using DeepFace and compares them with approved images for face recognition and verification.

## Features

- **Face Embedding Extraction**: Extract high-dimensional face embeddings using state-of-the-art DeepFace models
- **Face Quality Assessment**: Verify image quality before embedding extraction
- **Similarity Comparison**: Compare face embeddings using multiple distance metrics
- **Batch Processing**: Process multiple images and extract embeddings from directories
- **Embedding Storage**: Save and load embeddings in JSON format
- **Approval System**: Compare captured images with approved reference images

## AI Models Used

The system uses **DeepFace** library with the following AI models:

1. **VGG-Face** (Default): High accuracy, good for general face recognition
2. **Facenet**: Google's face recognition model, very accurate
3. **OpenFace**: Lightweight model, good for real-time applications
4. **DeepID**: High accuracy for face verification
5. **ArcFace**: State-of-the-art face recognition model

## Distance Metrics

- **Cosine Similarity** (Default): Measures angle between vectors, good for face embeddings
- **Euclidean Distance**: Standard distance metric
- **Manhattan Distance**: City-block distance metric

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Download the face landmark predictor (if not already present):
```bash
# The shape_predictor_68_face_landmarks.dat file should already be in the Face directory
```

## Usage

### Basic Usage

1. **Extract embeddings from your face image**:
```bash
python extract_face_embeddings.py --image face_embeddings.jpg
```

2. **Compare with approved images**:
```bash
python extract_face_embeddings.py --image captured_image.jpg --approved-dir aproved_img
```

3. **Save embeddings for later use**:
```bash
python extract_face_embeddings.py --image face_embeddings.jpg --save-embeddings --output-dir embeddings
```

### Advanced Usage

1. **Use different AI model**:
```bash
python extract_face_embeddings.py --image face_embeddings.jpg --model Facenet
```

2. **Adjust similarity threshold**:
```bash
python extract_face_embeddings.py --image face_embeddings.jpg --threshold 0.7
```

3. **Use different distance metric**:
```bash
python extract_face_embeddings.py --image face_embeddings.jpg --metric euclidean
```

### Testing

Run the test script to verify everything works:
```bash
python test_embeddings.py
```

## File Structure

```
Face/
├── face_embedding_extractor.py    # Main embedding extraction class
├── extract_face_embeddings.py     # Command-line interface
├── test_embeddings.py             # Test script
├── aproved_img/                   # Directory for approved reference images
│   └── lastname_firstname_20250806_122711.jpg
├── embeddings/                    # Output directory for saved embeddings
├── face_embeddings.jpg           # Your input face image
└── requirements.txt              # Python dependencies
```

## API Usage

### Basic Python Usage

```python
from face_embedding_extractor import FaceEmbeddingExtractor

# Initialize extractor
extractor = FaceEmbeddingExtractor(model_name="VGG-Face", distance_metric="cosine")

# Extract embedding from image
embedding = extractor.extract_embedding("face_embeddings.jpg")

# Compare with approved images
results = extractor.compare_with_approved_images("captured_image.jpg", "aproved_img")

# Check if approved
if results['is_approved']:
    print("Face matches approved reference!")
else:
    print("Face does not match approved reference.")
```

### Extract Embeddings from Directory

```python
# Extract embeddings from all images in a directory
embeddings = extractor.extract_embeddings_from_directory("aproved_img")

# Save embeddings
extractor.save_embeddings(embeddings, "saved_embeddings.json")

# Load embeddings later
loaded_embeddings = extractor.load_embeddings("saved_embeddings.json")
```

## Configuration

### Similarity Threshold

- **0.5-0.6**: Strict matching (fewer false positives)
- **0.6-0.7**: Balanced matching (recommended)
- **0.7-0.8**: Loose matching (more false positives)

### Model Selection

- **VGG-Face**: Good balance of accuracy and speed
- **Facenet**: High accuracy, slower
- **OpenFace**: Fast, good for real-time
- **ArcFace**: Highest accuracy, requires more resources

## Output Format

The system provides detailed comparison results:

```json
{
  "test_image": "face_embeddings.jpg",
  "approved_dir": "aproved_img",
  "matches": [
    {
      "filename": "lastname_firstname_20250806_122711.jpg",
      "similarity_score": 0.8234,
      "is_match": true
    }
  ],
  "best_match": {
    "filename": "lastname_firstname_20250806_122711.jpg",
    "similarity_score": 0.8234
  },
  "is_approved": true,
  "timestamp": "2024-01-15T10:30:00"
}
```

## Troubleshooting

### Common Issues

1. **"No face detected" error**:
   - Ensure the image contains a clear, front-facing face
   - Check image quality and lighting
   - Try different AI models

2. **Low similarity scores**:
   - Adjust similarity threshold
   - Ensure consistent lighting conditions
   - Use higher quality reference images

3. **Installation issues**:
   - Install Visual Studio Build Tools (Windows)
   - Install cmake and dlib dependencies
   - Use conda environment for easier installation

### Performance Tips

- Use GPU acceleration if available
- Pre-extract and save embeddings for approved images
- Use appropriate model size for your use case
- Batch process multiple images for efficiency

## Integration with Existing System

The face embedding extractor can be integrated with your existing face detection system:

```python
from face_embedding_extractor import FaceEmbeddingExtractor
from face_detector_try import FaceDetector

# Initialize both systems
face_detector = FaceDetector()
embedding_extractor = FaceEmbeddingExtractor()

# Use face detector for quality checks
# Use embedding extractor for recognition
```

## Security Considerations

- Store embeddings securely (encrypted if needed)
- Use appropriate similarity thresholds for your security requirements
- Regularly update approved reference images
- Implement liveness detection to prevent spoofing attacks 