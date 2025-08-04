#!/usr/bin/env python3
"""
HTTPS Server for Phone Fingerprint Integration
This version uses HTTPS which is required for WebAuthn fingerprint access.
"""

import ssl
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import base64
import time
import threading
import os
from phone_scanner import PhoneFingerprintScanner
from phone_matcher import PhoneFingerprintMatcher
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for mobile app communication

class PhoneFingerprintAPI:
    def __init__(self, threshold: float = 0.75):
        self.scanner = PhoneFingerprintScanner()
        self.matcher = PhoneFingerprintMatcher(self.scanner, threshold)
        self.last_activity = time.time()
        
    def process_phone_data(self, phone_data: dict):
        """Process fingerprint data received from phone"""
        try:
            logger.info(f"📱 Received data from phone: {phone_data['finger_id']}")
            
            # Decode the raw sensor data
            raw_data = base64.b64decode(phone_data['raw_data'])
            
            # Process the fingerprint data
            result = self.scanner.receive_fingerprint_from_phone(
                phone_data['finger_id'],
                raw_data,
                phone_data['sensor_type'],
                phone_data['device_info']
            )
            
            if result:
                logger.info(f"✅ Successfully processed {phone_data['finger_id']}")
                return {
                    "success": True,
                    "message": f"Fingerprint {phone_data['finger_id']} processed successfully",
                    "quality_score": result.quality_score,
                    "timestamp": time.time()
                }
            else:
                logger.error(f"❌ Failed to process {phone_data['finger_id']}")
                return {
                    "success": False,
                    "message": "Failed to process fingerprint data",
                    "timestamp": time.time()
                }
                
        except Exception as e:
            logger.error(f"❌ Error processing phone data: {e}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "timestamp": time.time()
            }
    
    def authenticate_fingerprint(self, phone_data: dict):
        """Authenticate fingerprint data from phone"""
        try:
            logger.info(f"🔐 Authenticating {phone_data['finger_id']}")
            
            # Decode the raw sensor data
            raw_data = base64.b64decode(phone_data['raw_data'])
            
            # Create test data object
            test_data = self.scanner.receive_fingerprint_from_phone(
                phone_data['finger_id'],
                raw_data,
                phone_data['sensor_type'],
                phone_data['device_info']
            )
            
            if not test_data:
                return {
                    "success": False,
                    "message": "Failed to process authentication data",
                    "timestamp": time.time()
                }
            
            # Authenticate against registered fingerprints
            result = self.matcher.authenticate_fingerprint(phone_data['finger_id'], test_data)
            
            logger.info(f"🔍 Authentication result: {'✅ MATCH' if result.is_match else '❌ NO MATCH'}")
            
            return {
                "success": True,
                "is_match": result.is_match,
                "similarity_score": result.similarity_score,
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "message": "Authentication successful" if result.is_match else "Authentication failed",
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Error during authentication: {e}")
            return {
                "success": False,
                "message": f"Authentication error: {str(e)}",
                "timestamp": time.time()
            }

# Initialize the API system
api_system = PhoneFingerprintAPI(threshold=0.75)

@app.route('/')
def home():
    """Home page with API documentation"""
    server_url = request.host_url.rstrip('/')
    registered_count = len(api_system.scanner.scanned_fingers)
    uptime = time.strftime('%H:%M:%S', time.gmtime(time.time() - api_system.last_activity))
    last_activity = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(api_system.last_activity))
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>🔐 HTTPS Phone Fingerprint Server</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; color: #333; margin-bottom: 30px; }}
            .status {{ padding: 15px; border-radius: 5px; margin: 10px 0; }}
            .status.success {{ background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }}
            .status.error {{ background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }}
            .status.info {{ background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }}
            .button {{ background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; text-decoration: none; display: inline-block; }}
            .button:hover {{ background: #0056b3; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔐 HTTPS Phone Fingerprint Server</h1>
                <p>Secure server for real fingerprint sensor access</p>
            </div>
            
            <div class="status success">
                <strong>✅ Server Status:</strong> Running securely on {server_url}
            </div>
            
            <h2>📱 Phone Apps</h2>
            <div class="status info">
                <p><strong>For real fingerprint sensor access:</strong></p>
                <a href="/real_fingerprint_app.html" class="button">🔐 Real Fingerprint App (HTTPS)</a>
                <a href="/mobile_app_example.html" class="button">📱 Simulated App</a>
            </div>
            
            <h2>📊 System Information</h2>
            <div class="status info">
                <p><strong>Registered Fingers:</strong> {registered_count}</p>
                <p><strong>Server Uptime:</strong> {uptime}</p>
                <p><strong>Last Activity:</strong> {last_activity}</p>
            </div>
            
            <h2>🔧 API Endpoints</h2>
            <div class="status info">
                <p><strong>Register:</strong> {server_url}/api/register</p>
                <p><strong>Authenticate:</strong> {server_url}/api/authenticate</p>
                <p><strong>Status:</strong> {server_url}/api/status</p>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/real_fingerprint_app.html')
def real_fingerprint_app():
    """Serve the real fingerprint app that uses WebAuthn"""
    try:
        # Look for the file in the current directory (phone_fingerprint folder)
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'real_fingerprint_app.html')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Update the server URL to use HTTPS
        content = content.replace('http://192.168.88.62:5000', 'https://192.168.88.62:5001')
        content = content.replace('http://localhost:5000', 'https://localhost:5001')
        
        return content
    except FileNotFoundError:
        return "Real fingerprint app not found", 404

@app.route('/mobile_app_example.html')
def mobile_app_example():
    """Serve the mobile app example"""
    try:
        # Look for the file in the current directory (phone_fingerprint folder)
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'mobile_app_example.html')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Update the server URL to use HTTPS
        content = content.replace('http://192.168.88.62:5000', 'https://192.168.88.62:5001')
        content = content.replace('http://localhost:5000', 'https://localhost:5001')
        
        return content
    except FileNotFoundError:
        return "Mobile app example not found", 404

@app.route('/api/register', methods=['POST'])
def register_fingerprint():
    """Register a new fingerprint from phone"""
    try:
        phone_data = request.get_json()
        if not phone_data:
            return jsonify({"success": False, "message": "No data received"}), 400
        
        logger.info(f"📝 Registration request for {phone_data.get('finger_id', 'unknown')}")
        
        result = api_system.process_phone_data(phone_data)
        api_system.last_activity = time.time()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Registration error: {e}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

@app.route('/api/authenticate', methods=['POST'])
def authenticate_fingerprint():
    """Authenticate fingerprint from phone"""
    try:
        phone_data = request.get_json()
        if not phone_data:
            return jsonify({"success": False, "message": "No data received"}), 400
        
        logger.info(f"🔐 Authentication request for {phone_data.get('finger_id', 'unknown')}")
        
        result = api_system.authenticate_fingerprint(phone_data)
        api_system.last_activity = time.time()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"❌ Authentication error: {e}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    try:
        stats = api_system.matcher.get_matching_statistics()
        
        status = {
            "success": True,
            "server_running": True,
            "registered_fingers": len(api_system.scanner.scanned_fingers),
            "finger_names": list(api_system.scanner.scanned_fingers),
            "last_activity": api_system.last_activity,
            "uptime": time.time() - api_system.last_activity,
            "statistics": stats if "error" not in stats else None
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"❌ Status error: {e}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

def create_self_signed_cert():
    """Create a self-signed certificate for HTTPS"""
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "CA"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Phone Fingerprint"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        # Save certificate and key
        cert_path = "cert.pem"
        key_path = "key.pem"
        
        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        with open(key_path, "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        return cert_path, key_path
        
    except ImportError:
        print("⚠️  cryptography library not found. Using basic SSL.")
        return None, None

def start_https_server(host='0.0.0.0', port=5001, debug=True):
    """Start the HTTPS Flask server"""
    print("="*60)
    print("🔐 HTTPS PHONE FINGERPRINT SERVER")
    print("="*60)
    print(f"🌐 Server URL: https://{host}:{port}")
    print(f"📱 Phone Integration: Ready for real fingerprint sensor")
    print(f"💻 Computer Processing: All processing happens on this device")
    print("="*60)
    print()
    print("📋 Available Endpoints:")
    print(f"   Home: https://{host}:{port}/")
    print(f"   Real Fingerprint: https://{host}:{port}/real_fingerprint_app.html")
    print(f"   Simulated: https://{host}:{port}/mobile_app_example.html")
    print()
    print("📱 To connect your phone:")
    print("   1. Make sure phone and computer are on same network")
    print("   2. Use the HTTPS URL in your mobile browser")
    print("   3. Accept the security certificate warning")
    print("   4. Your fingerprint sensor should now work!")
    print()
    
    # Create SSL context
    cert_path, key_path = create_self_signed_cert()
    
    if cert_path and key_path:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_path, key_path)
        app.run(host=host, port=port, debug=debug, ssl_context=context)
    else:
        # Fallback to basic SSL
        app.run(host=host, port=port, debug=debug, ssl_context='adhoc')

if __name__ == '__main__':
    start_https_server() 