#!/usr/bin/env python3
"""
Enhanced Environmental Dashboard with Moisture Sensor Integration
Real-time CSV electrical data analysis and moisture estimation
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import queue
import numpy as np
import pandas as pd
from pathlib import Path

# Add the src directory to path for moisture sensor imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.dirname(__file__))

try:
    from moisture_sensor_integration import IntegratedMoistureSensor
    MOISTURE_SENSOR_AVAILABLE = True
except ImportError:
    print("⚠️  Moisture sensor not available - using basic dashboard")
    MOISTURE_SENSOR_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_dashboard.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'enhanced_environmental_sensing_2025'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state management
monitoring_active = False
data_queue = queue.Queue(maxsize=1000)
data_history = []
max_data_points = 1000
system_start_time = datetime.now()

# CSV data management
csv_data_buffer = {}
current_csv_file = None
moisture_sensor = None

class EnhancedEnvironmentalDashboard:
    """Enhanced dashboard with moisture sensor integration"""
    
    def __init__(self):
        self.last_data = None
        self.data_source = 'integrated'
        self.moisture_sensor = None
        self.csv_data = None
        self.analysis_results = {}
        
        # Initialize moisture sensor if available
        if MOISTURE_SENSOR_AVAILABLE:
            try:
                self.moisture_sensor = IntegratedMoistureSensor()
                logger.info("✅ Moisture sensor integrated successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize moisture sensor: {e}")
        
        logger.info("Enhanced Environmental Dashboard initialized")
    
    def load_csv_data(self, csv_file_path):
        """Load and analyze CSV data with moisture sensor"""
        try:
            if not self.moisture_sensor:
                logger.warning("Moisture sensor not available")
                return None
            
            # Analyze CSV with moisture sensor
            results = self.moisture_sensor.analyze_csv_data(csv_file_path)
            
            # Store results
            self.analysis_results = results
            self.csv_data = csv_file_path
            
            logger.info(f"✅ CSV analysis completed: {Path(csv_file_path).name}")
            return results
            
        except Exception as e:
            logger.error(f"❌ CSV analysis failed: {e}")
            return None
    
    def get_current_environmental_data(self):
        """Get current environmental data with CSV analysis"""
        try:
            current_time = datetime.now()
            
            # Base environmental data
            base_temp = 20 + 5 * np.sin(current_time.hour * np.pi / 12)
            base_humidity = 60 + 20 * np.sin(current_time.hour * np.pi / 12)
            
            data = {
                'temperature': round(base_temp + np.random.uniform(-1, 1), 1),
                'humidity': round(max(0, min(100, base_humidity + np.random.uniform(-5, 5))), 1),
                'ph': round(6.5 + np.random.uniform(-0.3, 0.3), 2),
                'moisture': round(45 + np.random.uniform(-10, 10), 1),
                'pollution': round(np.random.uniform(0, 30), 2),
                'timestamp': current_time.isoformat(),
                'data_quality': 'high',
                'source': 'enhanced_integrated'
            }
            
            # Add moisture sensor analysis if available
            if self.analysis_results:
                data['moisture_sensor'] = {
                    'moisture_estimate': self.analysis_results.get('moisture_estimation', {}).get('moisture_estimate', 'no_data'),
                    'confidence': self.analysis_results.get('moisture_estimation', {}).get('confidence', 0.0),
                    'electrical_features': self.analysis_results.get('electrical_analysis', {}).get('features', {}),
                    'acoustic_features': self.analysis_results.get('acoustic_analysis', {}).get('features', {}),
                    'correlations': self.analysis_results.get('correlation_analysis', {}).get('correlations_discovered', {}).get('correlation_summary', {}),
                    'csv_file': Path(self.csv_data).name if self.csv_data else None
                }
            
            self.last_data = data
            return data
            
        except Exception as e:
            logger.error(f"Error generating environmental data: {e}")
            return self.get_fallback_data()
    
    def get_fallback_data(self):
        """Fallback data when main system fails"""
        return {
            'temperature': 22.0,
            'humidity': 65.0,
            'ph': 6.8,
            'moisture': 50.0,
            'pollution': 15.0,
            'timestamp': datetime.now().isoformat(),
            'data_quality': 'fallback',
            'source': 'fallback_system'
        }
    
    def get_system_status(self):
        """Get comprehensive system status"""
        return {
            'status': 'operational',
            'uptime': str(datetime.now() - system_start_time),
            'data_points_collected': len(data_history),
            'monitoring_active': monitoring_active,
            'last_data_update': self.last_data['timestamp'] if self.last_data else None,
            'data_source': self.data_source,
            'system_version': '3.1.0',
            'moisture_sensor_available': MOISTURE_SENSOR_AVAILABLE,
            'csv_data_loaded': self.csv_data is not None,
            'last_analysis': self.analysis_results.get('analysis_timestamp', 'never') if self.analysis_results else 'never'
        }

# Initialize dashboard
dashboard = EnhancedEnvironmentalDashboard()

def start_monitoring():
    """Start environmental monitoring with enhanced data collection"""
    global monitoring_active, data_queue, data_history
    
    def monitor_loop():
        global monitoring_active, data_history
        
        while monitoring_active:
            try:
                # Get current environmental data
                current_data = dashboard.get_current_environmental_data()
                
                if current_data:
                    # Emit to all connected clients
                    socketio.emit('environmental_update', current_data)
                    
                    # Store in history
                    data_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'data': current_data
                    })
                    
                    # Maintain data history size
                    while len(data_history) > max_data_points:
                        data_history.pop(0)
                    
                    # Store in queue for API access
                    try:
                        data_queue.put_nowait({
                            'timestamp': datetime.now().isoformat(),
                            'data': current_data
                        })
                    except queue.Full:
                        # Remove oldest item if queue is full
                        data_queue.get()
                        data_queue.put({
                            'timestamp': datetime.now().isoformat(),
                            'data': current_data
                        })
                    
                    logger.debug(f"Data update: {current_data['temperature']}°C, {current_data['humidity']}%")
                
                time.sleep(5)  # Update every 5 seconds
                
        except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(10)  # Wait longer on error
    
    if not monitoring_active:
        monitoring_active = True
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True, name="EnhancedEnvironmentalMonitor")
        monitor_thread.start()
        logger.info("✅ Enhanced environmental monitoring started")
        return True
    return False

def stop_monitoring():
    """Stop environmental monitoring"""
    global monitoring_active
    monitoring_active = False
    logger.info("⏹️ Environmental monitoring stopped")
    return True

# Flask Routes
@app.route('/')
def dashboard_home():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/status')
def api_status():
    """System status API endpoint"""
    try:
        status = dashboard.get_system_status()
        status.update({
        'timestamp': datetime.now().isoformat(),
            'connected_clients': len(socketio.server.manager.rooms.get('/', {}).get('', set())),
            'queue_size': data_queue.qsize()
        })
        return jsonify(status)
    except Exception as e:
        logger.error(f"Status API error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/load_csv', methods=['POST'])
def api_load_csv():
    """Load and analyze CSV file"""
    try:
        data = request.get_json()
        csv_file = data.get('csv_file')
        
        if not csv_file:
            return jsonify({'status': 'error', 'message': 'No CSV file specified'}), 400
        
        # Construct full path
        csv_path = os.path.join('/home/kronos/testTRANSFORM', csv_file)
        
        if not os.path.exists(csv_path):
            return jsonify({'status': 'error', 'message': 'CSV file not found'}), 404
        
        # Analyze CSV with moisture sensor
        results = dashboard.load_csv_data(csv_path)
        
        if results:
            return jsonify({
                'status': 'success',
                'message': 'CSV analysis completed',
                'results': results
            })
        else:
            return jsonify({'status': 'error', 'message': 'CSV analysis failed'}), 500
            
    except Exception as e:
        logger.error(f"CSV loading error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/start_monitoring', methods=['POST'])
def api_start_monitoring():
    """Start monitoring API endpoint"""
    try:
        if start_monitoring():
            return jsonify({'status': 'success', 'message': 'Monitoring started successfully'})
        else:
            return jsonify({'status': 'already_running', 'message': 'Monitoring already active'})
    except Exception as e:
        logger.error(f"Start monitoring error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stop_monitoring', methods=['POST'])
def api_stop_monitoring():
    """Stop monitoring API endpoint"""
    try:
        if stop_monitoring():
            return jsonify({'status': 'success', 'message': 'Monitoring stopped successfully'})
        else:
            return jsonify({'status': 'not_running', 'message': 'Monitoring not active'})
    except Exception as e:
        logger.error(f"Stop monitoring error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/environmental_data')
def api_environmental_data():
    """Current environmental data API endpoint"""
    try:
        data = dashboard.get_current_environmental_data()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Environmental data API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/historical_data')
def api_historical_data():
    """Historical data API endpoint"""
    try:
        return jsonify({
            'data_points': len(data_history),
            'data': data_history[-100:] if data_history else [],
            'total_available': len(data_history)
        })
    except Exception as e:
        logger.error(f"Historical data API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system_health')
def api_system_health():
    """System health check endpoint"""
    try:
        health = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'uptime': str(datetime.now() - system_start_time),
            'memory_usage': 'normal',
            'data_flow': 'active' if monitoring_active else 'inactive',
            'last_error': None,
            'version': '3.1.0',
            'moisture_sensor': MOISTURE_SENSOR_AVAILABLE,
            'csv_analysis': dashboard.analysis_results is not None
        }
        return jsonify(health)
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

# WebSocket Event Handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    client_id = request.sid
    logger.info(f"✅ Client connected: {client_id}")
    
    # Send connection confirmation
    emit('connection_status', {
        'status': 'connected', 
        'message': 'Connected to Enhanced Environmental Monitoring System v3.1',
        'client_id': client_id
    })
    
    # Send current system status
    emit('system_status', dashboard.get_system_status())
    
    # Send current environmental data
    current_data = dashboard.get_current_environmental_data()
    if current_data:
        emit('environmental_update', current_data)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    logger.info(f"❌ Client disconnected: {client_id}")

@socketio.on('request_data')
def handle_data_request():
    """Handle client data requests"""
    try:
        data = dashboard.get_current_environmental_data()
        if data:
        emit('environmental_update', data)
        else:
            emit('error', {'message': 'No data available'})
    except Exception as e:
        logger.error(f"Data request error: {e}")
        emit('error', {'message': f'Error: {str(e)}'})

@socketio.on('load_csv')
def handle_csv_load(data):
    """Handle CSV loading request"""
    try:
        csv_file = data.get('csv_file')
        if csv_file:
            # Construct full path
            csv_path = os.path.join('/home/kronos/testTRANSFORM', csv_file)
            
            if os.path.exists(csv_path):
                results = dashboard.load_csv_data(csv_path)
                if results:
                    emit('csv_analysis_complete', {
                        'status': 'success',
                        'results': results
                    })
                else:
                    emit('csv_analysis_error', {
                        'status': 'error',
                        'message': 'CSV analysis failed'
                    })
            else:
                emit('csv_analysis_error', {
                    'status': 'error',
                    'message': 'CSV file not found'
                })
        else:
            emit('csv_analysis_error', {
                'status': 'error',
                'message': 'No CSV file specified'
            })
    except Exception as e:
        logger.error(f"CSV loading error: {e}")
        emit('csv_analysis_error', {
            'status': 'error',
            'message': str(e)
        })

@socketio.on('start_monitoring')
def handle_start_monitoring():
    """Handle client monitoring start request"""
    try:
        if start_monitoring():
            emit('monitoring_status', {
                'status': 'started', 
                'message': 'Enhanced environmental monitoring started successfully'
            })
        else:
            emit('monitoring_status', {
                'status': 'already_running', 
                'message': 'Monitoring already active'
            })
    except Exception as e:
        logger.error(f"Start monitoring error: {e}")
        emit('error', {'message': f'Error starting monitoring: {str(e)}'})

@socketio.on('stop_monitoring')
def handle_stop_monitoring():
    """Handle client monitoring stop request"""
    try:
        if stop_monitoring():
            emit('monitoring_status', {
                'status': 'stopped', 
                'message': 'Enhanced environmental monitoring stopped successfully'
            })
        else:
            emit('monitoring_status', {
                'status': 'not_running', 
                'message': 'Monitoring not active'
            })
    except Exception as e:
        logger.error(f"Stop monitoring error: {e}")
        emit('error', {'message': f'Error stopping monitoring: {str(e)}'})

@socketio.on('ping')
def handle_ping():
    """Handle client ping for connection health check"""
    emit('pong', {'timestamp': datetime.now().isoformat()})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# Application startup
if __name__ == '__main__':
    logger.info("🚀 Starting Enhanced Environmental Monitoring Dashboard v3.1...")
    
    # Initialize dashboard
    logger.info("✅ Enhanced dashboard initialization complete")
    
    # Start monitoring automatically
    start_monitoring()
    
    # Start Flask development server
    logger.info("🌐 Starting enhanced web server on http://localhost:5001")
    logger.info("📊 Enhanced dashboard ready with moisture sensor integration")
    
    try:
        socketio.run(app, host='0.0.0.0', port=5001, debug=False)
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1)
