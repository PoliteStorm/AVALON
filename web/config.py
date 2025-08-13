#!/usr/bin/env python3
"""
Web Dashboard Configuration
Configuration settings for the environmental monitoring web dashboard
"""

import os
from datetime import timedelta

class Config:
    """Base configuration class"""
    
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'environmental_sensing_2025_dev_key'
    DEBUG = False
    TESTING = False
    
    # Server configuration
    HOST = '0.0.0.0'
    PORT = 5000
    
    # WebSocket configuration
    SOCKETIO_ASYNC_MODE = 'threading'
    SOCKETIO_PING_TIMEOUT = 10
    SOCKETIO_PING_INTERVAL = 25
    SOCKETIO_MAX_HTTP_BUFFER_SIZE = 1e6
    
    # Data configuration
    MAX_DATA_POINTS = 1000
    UPDATE_INTERVAL = 5  # seconds
    HEARTBEAT_INTERVAL = 30  # seconds
    
    # Chart configuration
    CHART_UPDATE_INTERVAL = 1  # seconds
    CHART_MAX_POINTS = 100
    CHART_ANIMATION_DURATION = 750
    
    # Export configuration
    EXPORT_FORMATS = ['json', 'csv', 'svg', 'html']
    EXPORT_MAX_RECORDS = 10000
    
    # Security configuration
    SESSION_COOKIE_SECURE = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Logging configuration
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = 'web_dashboard.log'
    
    # Database configuration (if needed in future)
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///environmental_dashboard.db'
    
    # API configuration
    API_RATE_LIMIT = '100 per minute'
    API_TIMEOUT = 30
    
    # File upload configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'csv', 'json', 'txt', 'xlsx'}
    
    # Email configuration (for alerts)
    MAIL_SERVER = os.environ.get('MAIL_SERVER') or 'localhost'
    MAIL_PORT = int(os.environ.get('MAIL_PORT') or 587)
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    
    # Monitoring configuration
    ENABLE_MONITORING = True
    MONITORING_INTERVAL = 5  # seconds
    ALERT_THRESHOLDS = {
        'temperature': {'min': -10, 'max': 50, 'critical': [-20, 60]},
        'humidity': {'min': 0, 'max': 100, 'critical': [0, 100]},
        'ph': {'min': 0, 'max': 14, 'critical': [0, 14]},
        'moisture': {'min': 0, 'max': 100, 'critical': [0, 100]},
        'pollution': {'min': 0, 'max': 1000, 'critical': [100, 1000]}
    }
    
    # Performance configuration
    ENABLE_CACHING = True
    CACHE_TIMEOUT = 300  # seconds
    MAX_CONCURRENT_USERS = 100
    
    # Feature flags
    ENABLE_REAL_TIME_UPDATES = True
    ENABLE_DATA_EXPORT = True
    ENABLE_CHART_CUSTOMIZATION = True
    ENABLE_ALERT_SYSTEM = True
    ENABLE_USER_AUTHENTICATION = False  # Enable in production
    
    # UI configuration
    THEME = 'light'  # light, dark, auto
    LANGUAGE = 'en'
    TIMEZONE = 'UTC'
    
    # Chart colors and styling
    CHART_COLORS = {
        'temperature': '#198754',
        'humidity': '#0dcaf0',
        'ph': '#ffc107',
        'moisture': '#6c757d',
        'pollution': '#dc3545'
    }
    
    # Environmental parameter ranges
    PARAMETER_RANGES = {
        'temperature': {'min': -20, 'max': 60, 'unit': '°C'},
        'humidity': {'min': 0, 'max': 100, 'unit': '%'},
        'ph': {'min': 0, 'max': 14, 'unit': ''},
        'moisture': {'min': 0, 'max': 100, 'unit': '%'},
        'pollution': {'min': 0, 'max': 1000, 'unit': 'ppm'}
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    
    DEBUG = True
    TESTING = False
    
    # Development-specific settings
    HOST = 'localhost'
    PORT = 5000
    
    # Enable detailed logging
    LOG_LEVEL = 'DEBUG'
    
    # Development features
    ENABLE_DEBUG_TOOLBAR = True
    ENABLE_RELOADER = True
    
    # Less strict security for development
    SESSION_COOKIE_SECURE = False
    
    # Development data
    MOCK_DATA_ENABLED = True
    MOCK_UPDATE_INTERVAL = 2  # seconds

class TestingConfig(Config):
    """Testing configuration"""
    
    TESTING = True
    DEBUG = False
    
    # Test-specific settings
    WTF_CSRF_ENABLED = False
    PRESERVE_CONTEXT_ON_EXCEPTION = False
    
    # Use test database
    DATABASE_URL = 'sqlite:///:memory:'
    
    # Disable external services
    ENABLE_MONITORING = False
    ENABLE_EMAIL = False
    
    # Fast test intervals
    UPDATE_INTERVAL = 1
    HEARTBEAT_INTERVAL = 5

class ProductionConfig(Config):
    """Production configuration"""
    
    DEBUG = False
    TESTING = False
    
    # Production security
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    
    # Production logging
    LOG_LEVEL = 'WARNING'
    
    # Production features
    ENABLE_DEBUG_TOOLBAR = False
    ENABLE_RELOADER = False
    
    # Production monitoring
    ENABLE_MONITORING = True
    MONITORING_INTERVAL = 10  # seconds
    
    # Production performance
    ENABLE_CACHING = True
    MAX_CONCURRENT_USERS = 1000

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config(config_name=None):
    """Get configuration class by name"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    return config.get(config_name, config['default'])

def get_config_value(key, default=None):
    """Get a specific configuration value"""
    config_class = get_config()
    return getattr(config_class, key, default)

# Environment-specific configuration
if __name__ == '__main__':
    # Print current configuration
    current_config = get_config()
    print(f"Current configuration: {current_config.__name__}")
    print(f"Debug mode: {current_config.DEBUG}")
    print(f"Host: {current_config.HOST}")
    print(f"Port: {current_config.PORT}")
    print(f"Update interval: {current_config.UPDATE_INTERVAL}s") 