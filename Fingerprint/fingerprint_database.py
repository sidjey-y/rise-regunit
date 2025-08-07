import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, Float, DateTime, Text, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from base_processor import BaseProcessor
from config_manager import ConfigManager
import logging
from datetime import datetime

# SQLAlchemy Base
Base = declarative_base()

class FingerprintRecord(Base):
    """SQLAlchemy model for fingerprint records"""
    __tablename__ = 'fingerprints'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    subject_id = Column(String(50), nullable=False, index=True)
    finger_type = Column(String(20), nullable=False, index=True)
    hand_side = Column(String(10), nullable=False, index=True)
    minutiae_data = Column(Text, nullable=False)  # Store minutiae points as JSON
    fingerprint_metadata = Column(Text)  # Changed from 'metadata' to 'fingerprint_metadata'
    quality_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class FingerprintDatabase(BaseProcessor):
    """
    Database manager for fingerprint storage and retrieval using PostgreSQL.
    Handles fingerprint embeddings, metadata, and duplicate checking.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        super().__init__(config_manager)
        
        # Initialize attributes
        self.db_type = None
        self.db_config = {}
        self.connection_pool = None
        self.engine = None
        self.SessionLocal = None
        self.similarity_threshold = None
        self.max_similarity_score = None
        self.cache = {}
        self.cache_size = None
        
    def _initialize_components(self) -> None:
        """Initialize database components"""
        db_config = self.config_manager.get_database_config()
        verification_config = self.config_manager.get_verification_config()
        
        # Set configuration parameters
        self.db_type = db_config.get('type', 'postgresql')
        self.db_config = {
            'host': db_config.get('host', 'localhost'),
            'port': db_config.get('port', 5432),
            'database': db_config.get('database', 'fingerprint_db'),
            'username': db_config.get('username', 'fingerprint_user'),
            'password': db_config.get('password', 'your_secure_password'),
            'table_name': db_config.get('table_name', 'fingerprints'),
            'connection_pool_size': db_config.get('connection_pool_size', 10),
            'max_connections': db_config.get('max_connections', 20)
        }
        
        self.similarity_threshold = db_config.get('similarity_threshold', 0.85)
        self.max_similarity_score = db_config.get('max_similarity_score', 0.95)
        self.cache_size = verification_config.get('cache_size', 1000)
        
        # Initialize database connection
        self._create_database_connection()
        
        self.logger.info("PostgreSQL fingerprint database components initialized")
    
    def _create_database_connection(self) -> None:
        """Create PostgreSQL database connection and tables"""
        try:
            # Create SQLAlchemy engine with connection pooling
            connection_string = f"postgresql://{self.db_config['username']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            
            self.engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=self.db_config['connection_pool_size'],
                max_overflow=self.db_config['max_connections'] - self.db_config['connection_pool_size'],
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Create tables
            Base.metadata.create_all(bind=self.engine)
            
            # Create indexes for better performance
            self._create_indexes()
            
            self.logger.info(f"PostgreSQL database connected: {self.db_config['database']}")
            
        except Exception as e:
            self.logger.error(f"Error creating database connection: {e}")
            raise
    
    def _create_indexes(self) -> None:
        """Create database indexes for better performance"""
        try:
            with self.engine.connect() as conn:
                # Create indexes if they don't exist
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_subject_id ON fingerprints (subject_id)",
                    "CREATE INDEX IF NOT EXISTS idx_finger_type ON fingerprints (finger_type)",
                    "CREATE INDEX IF NOT EXISTS idx_hand_side ON fingerprints (hand_side)",
                    "CREATE INDEX IF NOT EXISTS idx_quality_score ON fingerprints (quality_score)",
                    "CREATE INDEX IF NOT EXISTS idx_created_at ON fingerprints (created_at)",
                    "CREATE INDEX IF NOT EXISTS idx_subject_finger_hand ON fingerprints (subject_id, finger_type, hand_side)"
                ]
                
                for index_sql in indexes:
                    conn.execute(text(index_sql))
                
                conn.commit()
                
            self.logger.info("Database indexes created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating indexes: {e}")
            # Don't raise error as indexes are optional for functionality
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process fingerprint data for database operations.
        
        Args:
            data: Dictionary containing operation type and fingerprint data
            
        Returns:
            dict: Operation results
        """
        operation = data.get('operation', 'check_duplicate')
        
        if operation == 'store':
            return self._store_fingerprint(data)
        elif operation == 'check_duplicate':
            return self._check_duplicate(data)
        elif operation == 'retrieve':
            return self._retrieve_fingerprint(data)
        elif operation == 'delete':
            return self._delete_fingerprint(data)
        else:
            self.logger.error(f"Unknown operation: {operation}")
            return {}
    
    def _store_fingerprint(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store fingerprint in PostgreSQL database.
        
        Args:
            data: Dictionary containing fingerprint data
            
        Returns:
            dict: Storage results
        """
        try:
            subject_id = data.get('subject_id')
            finger_type = data.get('finger_type')
            hand_side = data.get('hand_side')
            minutiae = data.get('minutiae')
            metadata = data.get('metadata', {})
            quality_score = data.get('quality_score', 0.0)
            
            if not all([subject_id is not None, finger_type is not None, hand_side is not None, minutiae is not None]):
                self.logger.error("Missing required fingerprint data")
                return {'success': False, 'error': 'Missing required data'}
            
            # Convert minutiae to JSON string
            minutiae_json = json.dumps(minutiae)
            
            # Convert metadata to JSON string
            metadata_json = json.dumps(metadata)
            
            # Convert numpy float to Python float
            quality_score_float = float(quality_score) if hasattr(quality_score, 'item') else quality_score
            
            session = self.SessionLocal()
            try:
                # Check if fingerprint already exists
                existing_record = session.query(FingerprintRecord).filter(
                    FingerprintRecord.subject_id == subject_id,
                    FingerprintRecord.finger_type == finger_type,
                    FingerprintRecord.hand_side == hand_side
                ).first()
                
                if existing_record:
                    # Update existing record
                    existing_record.minutiae_data = minutiae_json
                    existing_record.fingerprint_metadata = metadata_json
                    existing_record.quality_score = quality_score_float
                    existing_record.updated_at = datetime.utcnow()
                    operation = 'updated'
                else:
                    # Insert new record
                    new_record = FingerprintRecord(
                        subject_id=subject_id,
                        finger_type=finger_type,
                        hand_side=hand_side,
                        minutiae_data=minutiae_json,
                        fingerprint_metadata=metadata_json,
                        quality_score=quality_score_float
                    )
                    session.add(new_record)
                    operation = 'inserted'
                
                session.commit()
                
                # Clear cache
                self._clear_cache()
                
                return {
                    'success': True,
                    'operation': operation,
                    'subject_id': subject_id,
                    'finger_type': finger_type,
                    'hand_side': hand_side
                }
                
            finally:
                session.close()
            
        except Exception as e:
            self.logger.error(f"Error storing fingerprint: {e}")
            return {'success': False, 'error': str(e)}
    
    def _check_duplicate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check for duplicate fingerprints in PostgreSQL database.
        
        Args:
            data: Dictionary containing fingerprint embedding
            
        Returns:
            dict: Duplicate check results
        """
        try:
            minutiae = data.get('minutiae')
            subject_id = data.get('subject_id')
            
            if minutiae is None:
                self.logger.error("Minutiae data is required for duplicate check")
                return {'success': False, 'error': 'Minutiae data required'}
            
            # Check cache first
            minutiae_str = json.dumps(minutiae, sort_keys=True)
            cache_key = f"duplicate_check_{hash(minutiae_str)}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            session = self.SessionLocal()
            try:
                # Get all embeddings from database
                records = session.query(FingerprintRecord).all()
                
                duplicates = []
                max_similarity = 0.0
                best_match = None
                
                for record in records:
                    # Skip if it's the same subject (unless checking for duplicates within same subject)
                    if subject_id and record.subject_id == subject_id:
                        continue
                    
                    # Parse stored minutiae data
                    try:
                        stored_minutiae = json.loads(record.minutiae_data)
                        if not stored_minutiae:
                            continue
                    except (json.JSONDecodeError, TypeError):
                        continue
                    
                    # Calculate similarity using minutiae comparison
                    similarity = self._compare_minutiae_similarity(minutiae, stored_minutiae)
                    
                    if similarity >= self.similarity_threshold:
                        duplicates.append({
                            'id': record.id,
                            'subject_id': record.subject_id,
                            'finger_type': record.finger_type,
                            'hand_side': record.hand_side,
                            'similarity': similarity,
                            'quality_score': record.quality_score
                        })
                        
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_match = duplicates[-1]
                
                # Sort duplicates by similarity (highest first)
                duplicates.sort(key=lambda x: x['similarity'], reverse=True)
                
                result = {
                    'success': True,
                    'is_duplicate': len(duplicates) > 0,
                    'duplicate_count': len(duplicates),
                    'max_similarity': max_similarity,
                    'best_match': best_match,
                    'all_duplicates': duplicates,
                    'threshold': self.similarity_threshold
                }
                
                # Cache result
                self._cache_result(cache_key, result)
                
                return result
                
            finally:
                session.close()
            
        except Exception as e:
            self.logger.error(f"Error checking duplicates: {e}")
            return {'success': False, 'error': str(e)}
    
    def _retrieve_fingerprint(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve fingerprint from PostgreSQL database.
        
        Args:
            data: Dictionary containing retrieval criteria
            
        Returns:
            dict: Retrieved fingerprint data
        """
        try:
            subject_id = data.get('subject_id')
            finger_type = data.get('finger_type')
            hand_side = data.get('hand_side')
            
            if not subject_id:
                self.logger.error("Subject ID is required for retrieval")
                return {'success': False, 'error': 'Subject ID required'}
            
            session = self.SessionLocal()
            try:
                if finger_type and hand_side:
                    # Retrieve specific fingerprint
                    records = session.query(FingerprintRecord).filter(
                        FingerprintRecord.subject_id == subject_id,
                        FingerprintRecord.finger_type == finger_type,
                        FingerprintRecord.hand_side == hand_side
                    ).all()
                else:
                    # Retrieve all fingerprints for subject
                    records = session.query(FingerprintRecord).filter(
                        FingerprintRecord.subject_id == subject_id
                    ).all()
                
                if not records:
                    return {
                        'success': True,
                        'found': False,
                        'message': 'No fingerprints found'
                    }
                
                fingerprints = []
                for record in records:
                    # Convert embedding back to numpy array
                    embedding = np.frombuffer(record.embedding, dtype=np.float32)
                    
                    # Parse metadata
                    metadata = json.loads(record.fingerprint_metadata) if record.fingerprint_metadata else {}
                    
                    fingerprints.append({
                        'id': record.id,
                        'subject_id': record.subject_id,
                        'finger_type': record.finger_type,
                        'hand_side': record.hand_side,
                        'embedding': embedding,
                        'metadata': metadata,
                        'quality_score': record.quality_score,
                        'created_at': record.created_at.isoformat() if record.created_at else None
                    })
                
                return {
                    'success': True,
                    'found': True,
                    'fingerprint_count': len(fingerprints),
                    'fingerprints': fingerprints
                }
                
            finally:
                session.close()
            
        except Exception as e:
            self.logger.error(f"Error retrieving fingerprint: {e}")
            return {'success': False, 'error': str(e)}
    
    def _delete_fingerprint(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delete fingerprint from PostgreSQL database.
        
        Args:
            data: Dictionary containing deletion criteria
            
        Returns:
            dict: Deletion results
        """
        try:
            subject_id = data.get('subject_id')
            finger_type = data.get('finger_type')
            hand_side = data.get('hand_side')
            
            if not subject_id:
                self.logger.error("Subject ID is required for deletion")
                return {'success': False, 'error': 'Subject ID required'}
            
            session = self.SessionLocal()
            try:
                if finger_type and hand_side:
                    # Delete specific fingerprint
                    result = session.query(FingerprintRecord).filter(
                        FingerprintRecord.subject_id == subject_id,
                        FingerprintRecord.finger_type == finger_type,
                        FingerprintRecord.hand_side == hand_side
                    ).delete()
                else:
                    # Delete all fingerprints for subject
                    result = session.query(FingerprintRecord).filter(
                        FingerprintRecord.subject_id == subject_id
                    ).delete()
                
                session.commit()
                
                # Clear cache
                self._clear_cache()
                
                return {
                    'success': True,
                    'deleted_count': result,
                    'subject_id': subject_id
                }
                
            finally:
                session.close()
            
        except Exception as e:
            self.logger.error(f"Error deleting fingerprint: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Normalize embeddings
        embedding1_norm = embedding1 / np.linalg.norm(embedding1)
        embedding2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1_norm, embedding2_norm)
        
        return max(0.0, min(1.0, similarity))
    
    def _compare_minutiae_similarity(self, minutiae1: List[Dict], minutiae2: List[Dict]) -> float:
        """
        Compare two sets of minutiae points and return similarity score.
        
        Args:
            minutiae1: First set of minutiae points
            minutiae2: Second set of minutiae points
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            if not minutiae1 or not minutiae2:
                return 0.0
            
            # Extract coordinates and angles
            points1 = [(m.get('x', 0), m.get('y', 0), m.get('theta', 0)) for m in minutiae1]
            points2 = [(m.get('x', 0), m.get('y', 0), m.get('theta', 0)) for m in minutiae2]
            
            # Count similarity (how many points are similar)
            count_similarity = min(len(points1), len(points2)) / max(len(points1), len(points2))
            
            # Spatial similarity calculation
            spatial_similarity = 0.0
            if len(points1) > 0 and len(points2) > 0:
                max_points = min(5, len(points1), len(points2))  # Limit to 5 points for performance
                total_similarity = 0
                comparisons = 0
                
                for p1 in points1[:max_points]:
                    max_similarity = 0.0
                    for p2 in points2[:max_points]:
                        # Calculate Euclidean distance
                        dist = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
                        
                        # Calculate angle difference (normalized)
                        angle_diff = abs(p1[2] - p2[2]) % 360
                        if angle_diff > 180:
                            angle_diff = 360 - angle_diff
                        
                        # Normalize angle difference to 0-1 scale
                        angle_similarity = 1 - (angle_diff / 180)
                        
                        # Combined similarity (spatial + angular)
                        combined_similarity = (1 - dist/200) * 0.7 + angle_similarity * 0.3
                        combined_similarity = max(0, combined_similarity)
                        
                        max_similarity = max(max_similarity, combined_similarity)
                    
                    total_similarity += max_similarity
                    comparisons += 1
                
                if comparisons > 0:
                    spatial_similarity = total_similarity / comparisons
            
            # Combined similarity score
            similarity = (count_similarity * 0.3 + spatial_similarity * 0.7)
            
            return min(1.0, max(0.0, similarity))
            
        except Exception as e:
            self.logger.error(f"Error comparing minutiae: {e}")
            return 0.0
    
    def _cache_result(self, key: str, result: Dict[str, Any]) -> None:
        """
        Cache a result for faster retrieval.
        
        Args:
            key: Cache key
            result: Result to cache
        """
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = result
    
    def _clear_cache(self) -> None:
        """Clear the cache"""
        self.cache.clear()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get PostgreSQL database statistics.
        
        Returns:
            dict: Database statistics
        """
        try:
            session = self.SessionLocal()
            try:
                # Get total count
                total_count = session.query(FingerprintRecord).count()
                
                # Get unique subjects
                unique_subjects = session.query(FingerprintRecord.subject_id).distinct().count()
                
                # Get average quality score
                avg_quality = session.query(FingerprintRecord.quality_score).all()
                avg_quality = sum([q[0] for q in avg_quality]) / len(avg_quality) if avg_quality else 0.0
                
                # Get finger type distribution
                finger_type_dist = {}
                finger_types = session.query(FingerprintRecord.finger_type).all()
                for ft in finger_types:
                    finger_type_dist[ft[0]] = finger_type_dist.get(ft[0], 0) + 1
                
                # Get hand side distribution
                hand_side_dist = {}
                hand_sides = session.query(FingerprintRecord.hand_side).all()
                for hs in hand_sides:
                    hand_side_dist[hs[0]] = hand_side_dist.get(hs[0], 0) + 1
                
                return {
                    'total_fingerprints': total_count,
                    'unique_subjects': unique_subjects,
                    'average_quality_score': round(avg_quality, 3),
                    'finger_type_distribution': finger_type_dist,
                    'hand_side_distribution': hand_side_dist,
                    'database_type': 'PostgreSQL',
                    'connection_pool_size': self.db_config['connection_pool_size']
                }
                
            finally:
                session.close()
            
        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {'error': str(e)}
    
    def cleanup(self) -> None:
        """Cleanup database connection"""
        if self.engine:
            self.engine.dispose()
        self._clear_cache()
        super().cleanup() 