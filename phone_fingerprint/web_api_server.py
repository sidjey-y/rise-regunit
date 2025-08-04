"""
Web API Server for Phone Fingerprint Integration
This Flask server receives fingerprint data from your phone's capacitive sensor
and processes it on your computer in real-time.
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import base64
import time
import threading
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

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>📱 Phone Fingerprint API Server</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; color: #333; margin-bottom: 30px; }
        .status { padding: 15px; border-radius: 5px; margin: 10px 0; }
        .status.success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .status.error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .status.info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        .endpoint { background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #007bff; }
        .code { background: #e9ecef; padding: 10px; border-radius: 3px; font-family: monospace; font-size: 14px; }
        .button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
        .button:hover { background: #0056b3; }
        .log { background: #f8f9fa; padding: 10px; border-radius: 5px; font-family: monospace; font-size: 12px; max-height: 200px; overflow-y: auto; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📱 Phone Fingerprint API Server</h1>
            <p>Server is running and ready to receive fingerprint data from your phone</p>
        </div>
        
        <div class="status info">
            <strong>Server Status:</strong> Running on {{ server_url }}
        </div>
        
        <h2>📡 API Endpoints</h2>
        
        <div class="endpoint">
            <h3>1. Register Fingerprint</h3>
            <p><strong>POST</strong> {{ server_url }}/api/register</p>
            <div class="code">
{
  "finger_id": "thumb_left",
  "sensor_type": "capacitive",
  "raw_data": "base64_encoded_sensor_data",
  "device_info": {
    "model": "Samsung Galaxy S21",
    "sensor": "capacitive",
    "os": "Android 12"
  },
  "quality_score": 0.85,
  "timestamp": 1234567890.123
}
            </div>
        </div>
        
        <div class="endpoint">
            <h3>2. Authenticate Fingerprint</h3>
            <p><strong>POST</strong> {{ server_url }}/api/authenticate</p>
            <div class="code">
{
  "finger_id": "thumb_left",
  "sensor_type": "capacitive",
  "raw_data": "base64_encoded_sensor_data",
  "device_info": {
    "model": "Samsung Galaxy S21",
    "sensor": "capacitive",
    "os": "Android 12"
  },
  "quality_score": 0.85,
  "timestamp": 1234567890.123
}
            </div>
        </div>
        
        <div class="endpoint">
            <h3>3. Get System Status</h3>
            <p><strong>GET</strong> {{ server_url }}/api/status</p>
        </div>
        
        <h2>📱 Phone Integration</h2>
        <div class="status info">
            <p><strong>To connect your phone:</strong></p>
            <ul>
                <li>Make sure your phone and computer are on the same network</li>
                <li>Use the server URL above in your mobile app</li>
                <li>Send fingerprint data using the POST endpoints</li>
                <li>All processing happens on your computer</li>
            </ul>
        </div>
        
        <h2>📊 System Information</h2>
        <div class="status info">
            <p><strong>Registered Fingers:</strong> {{ registered_count }}</p>
            <p><strong>Server Uptime:</strong> {{ uptime }}</p>
            <p><strong>Last Activity:</strong> {{ last_activity }}</p>
        </div>
        
        <h2>🔧 Test the API</h2>
        <button class="button" onclick="testRegister()">Test Registration</button>
        <button class="button" onclick="testAuthenticate()">Test Authentication</button>
        <button class="button" onclick="getStatus()">Get Status</button>
        
        <div id="log" class="log"></div>
    </div>
    
    <script>
        function log(message) {
            const logDiv = document.getElementById('log');
            const timestamp = new Date().toLocaleTimeString();
            logDiv.innerHTML += `[${timestamp}] ${message}\n`;
            logDiv.scrollTop = logDiv.scrollHeight;
        }
        
        function testRegister() {
            log('Testing registration endpoint...');
            fetch('/api/register', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    finger_id: 'test_thumb',
                    sensor_type: 'capacitive',
                    raw_data: 'dGVzdF9kYXRh', // base64 encoded "test_data"
                    device_info: {
                        model: 'Test Phone',
                        sensor: 'capacitive',
                        os: 'Android 12'
                    },
                    quality_score: 0.85,
                    timestamp: Date.now() / 1000
                })
            })
            .then(response => response.json())
            .then(data => {
                log(`Registration result: ${JSON.stringify(data)}`);
            })
            .catch(error => {
                log(`Error: ${error}`);
            });
        }
        
        function testAuthenticate() {
            log('Testing authentication endpoint...');
            fetch('/api/authenticate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    finger_id: 'test_thumb',
                    sensor_type: 'capacitive',
                    raw_data: 'dGVzdF9kYXRh',
                    device_info: {
                        model: 'Test Phone',
                        sensor: 'capacitive',
                        os: 'Android 12'
                    },
                    quality_score: 0.85,
                    timestamp: Date.now() / 1000
                })
            })
            .then(response => response.json())
            .then(data => {
                log(`Authentication result: ${JSON.stringify(data)}`);
            })
            .catch(error => {
                log(`Error: ${error}`);
            });
        }
        
        function getStatus() {
            log('Getting system status...');
            fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                log(`Status: ${JSON.stringify(data)}`);
            })
            .catch(error => {
                log(`Error: ${error}`);
            });
        }
        
        // Auto-refresh status every 5 seconds
        setInterval(() => {
            fetch('/api/status')
            .then(response => response.json())
            .then(data => {
                // Update status display
            })
            .catch(error => {
                console.log('Status update error:', error);
            });
        }, 5000);
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Home page with API documentation"""
    server_url = request.host_url.rstrip('/')
    registered_count = len(api_system.scanner.scanned_fingers)
    uptime = time.strftime('%H:%M:%S', time.gmtime(time.time() - api_system.last_activity))
    last_activity = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(api_system.last_activity))
    
    return render_template_string(HTML_TEMPLATE, 
                                server_url=server_url,
                                registered_count=registered_count,
                                uptime=uptime,
                                last_activity=last_activity)

@app.route('/real_fingerprint_app.html')
def real_fingerprint_app():
    """Serve the real fingerprint app that uses WebAuthn"""
    try:
        # Look for the file in the current directory (phone_fingerprint folder)
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'real_fingerprint_app.html')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
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
            return f.read()
    except FileNotFoundError:
        return "Mobile app example not found", 404

@app.route('/simple_fingerprint_test.html')
def simple_fingerprint_test():
    """Serve the simple fingerprint test app"""
    try:
        # Look for the file in the current directory (phone_fingerprint folder)
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'simple_fingerprint_test.html')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Simple fingerprint test not found", 404

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

@app.route('/api/reset', methods=['POST'])
def reset_registration():
    """Reset all fingerprint data"""
    try:
        api_system.scanner.reset_registration()
        api_system.last_activity = time.time()
        
        logger.info("🗑️ Registration reset")
        
        return jsonify({
            "success": True,
            "message": "All fingerprint data has been reset"
        })
        
    except Exception as e:
        logger.error(f"❌ Reset error: {e}")
        return jsonify({"success": False, "message": f"Error: {str(e)}"}), 500

def start_server(host='0.0.0.0', port=5000, debug=True):
    """Start the Flask server"""
    print("="*60)
    print("📱 PHONE FINGERPRINT API SERVER")
    print("="*60)
    print(f"🌐 Server URL: http://{host}:{port}")
    print(f"📱 Phone Integration: Ready to receive fingerprint data")
    print(f"💻 Computer Processing: All processing happens on this device")
    print("="*60)
    print()
    print("📋 Available Endpoints:")
    print(f"   Home: http://{host}:{port}/")
    print(f"   Register: http://{host}:{port}/api/register")
    print(f"   Authenticate: http://{host}:{port}/api/authenticate")
    print(f"   Status: http://{host}:{port}/api/status")
    print(f"   Reset: http://{host}:{port}/api/reset")
    print()
    print("📱 To connect your phone:")
    print("   1. Make sure phone and computer are on same network")
    print("   2. Use the server URL in your mobile app")
    print("   3. Send fingerprint data using POST requests")
    print("   4. See real-time processing on this computer")
    print()
    
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    start_server() 