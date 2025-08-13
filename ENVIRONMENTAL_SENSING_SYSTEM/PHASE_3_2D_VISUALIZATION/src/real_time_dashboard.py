#!/usr/bin/env python3
"""
📊 Real-time Dashboard Backend - Phase 3
=========================================

This module provides the backend for real-time environmental monitoring
dashboard with live data updates and interactive features.

Author: Environmental Sensing Research Team
Date: August 12, 2025
Version: 1.0.0
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('PHASE_3_2D_VISUALIZATION/real_time_dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class EnvironmentalParameter:
    """Data class for environmental parameter values."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    confidence: float
    source: str  # 'real' or 'simulated'
    status: str  # 'normal', 'warning', 'alert', 'critical'

@dataclass
class Alert:
    """Data class for environmental alerts."""
    id: str
    parameter: str
    level: str  # 'info', 'warning', 'alert', 'critical'
    message: str
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False

@dataclass
class SystemStatus:
    """Data class for system health status."""
    overall_health: float  # 0-100%
    data_quality: float    # 0-100%
    sensor_status: Dict[str, str]  # sensor_id -> status
    last_update: datetime
    uptime: timedelta
    active_alerts: int

class RealTimeDashboard:
    """
    Real-time dashboard backend for environmental monitoring.
    
    This class provides:
    - Live data streaming
    - Real-time parameter updates
    - Alert management
    - System health monitoring
    - Performance metrics
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the Real-time Dashboard.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self.data_buffer = {}
        self.alert_history = []
        self.active_alerts = []
        self.callbacks = []
        self.is_running = False
        self.start_time = datetime.now()
        
        # Initialize data sources
        self.data_sources = self._initialize_data_sources()
        
        # Performance tracking
        self.performance_metrics = {
            'total_updates': 0,
            'average_response_time': 0.0,
            'last_response_time': 0.0,
            'data_throughput': 0.0
        }
        
        logger.info("📊 Real-time Dashboard initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'update_frequency': 5.0,  # seconds
            'data_buffer_size': 1000,
            'alert_thresholds': {
                'temperature': {'warning': 2.0, 'alert': 5.0, 'critical': 10.0},
                'humidity': {'warning': 10.0, 'alert': 20.0, 'critical': 30.0},
                'ph': {'warning': 0.5, 'alert': 1.0, 'critical': 2.0},
                'pollution': {'warning': 0.1, 'alert': 0.5, 'critical': 1.0}
            },
            'performance_targets': {
                'max_response_time': 100,  # milliseconds
                'min_data_quality': 80.0,  # percentage
                'max_alert_latency': 5.0   # seconds
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Could not load config file: {e}")
        
        return default_config
    
    def _initialize_data_sources(self) -> Dict[str, Any]:
        """Initialize data sources for environmental monitoring."""
        data_sources = {}
        
        # Try to connect to Phase 1 and 2 data sources
        potential_sources = [
            '../PHASE_1_DATA_INFRASTRUCTURE/RESULTS/baseline_analysis/',
            '../PHASE_2_AUDIO_SYNTHESIS/results/',
            '../RESULTS/baseline_analysis/',
            '../hybrid_environmental_sensing_system.py'
        ]
        
        for source in potential_sources:
            if Path(source).exists():
                data_sources[source] = {
                    'type': 'file_system',
                    'status': 'available',
                    'last_access': datetime.now()
                }
        
        # Add simulated data source for testing
        data_sources['simulated'] = {
            'type': 'simulation',
            'status': 'active',
            'last_access': datetime.now()
        }
        
        logger.info(f"Initialized {len(data_sources)} data sources")
        return data_sources
    
    def register_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Register a callback function for real-time updates.
        
        Args:
            callback: Function to call with update data
        """
        self.callbacks.append(callback)
        logger.info(f"Registered callback: {callback.__name__}")
    
    def unregister_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """
        Unregister a callback function.
        
        Args:
            callback: Function to unregister
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.info(f"Unregistered callback: {callback.__name__}")
    
    def _notify_callbacks(self, update_data: Dict[str, Any]):
        """Notify all registered callbacks with update data."""
        for callback in self.callbacks:
            try:
                callback(update_data)
            except Exception as e:
                logger.error(f"Error in callback {callback.__name__}: {e}")
    
    def get_current_parameters(self) -> List[EnvironmentalParameter]:
        """
        Get current environmental parameter values.
        
        Returns:
            List of current environmental parameters
        """
        current_params = []
        
        try:
            # Try to get real data first
            real_data = self._get_real_data()
            if real_data:
                for param_name, param_data in real_data.items():
                    current_params.append(EnvironmentalParameter(
                        name=param_name,
                        value=param_data['value'],
                        unit=param_data['unit'],
                        timestamp=param_data['timestamp'],
                        confidence=param_data['confidence'],
                        source='real',
                        status=self._assess_parameter_status(param_name, param_data['value'])
                    ))
            
            # Fill in missing parameters with simulated data
            simulated_data = self._generate_simulated_data()
            existing_params = {p.name for p in current_params}
            
            for param_name, param_data in simulated_data.items():
                if param_name not in existing_params:
                    current_params.append(EnvironmentalParameter(
                        name=param_name,
                        value=param_data['value'],
                        unit=param_data['unit'],
                        timestamp=param_data['timestamp'],
                        confidence=param_data['confidence'],
                        source='simulated',
                        status=self._assess_parameter_status(param_name, param_data['value'])
                    ))
            
            # Update data buffer
            self._update_data_buffer(current_params)
            
        except Exception as e:
            logger.error(f"Error getting current parameters: {e}")
            # Return simulated data as fallback
            current_params = self._get_fallback_data()
        
        return current_params
    
    def _get_real_data(self) -> Optional[Dict[str, Any]]:
        """Attempt to get real environmental data from available sources."""
        for source_path, source_info in self.data_sources.items():
            if source_info['status'] == 'available':
                try:
                    # Try to read recent data files
                    if source_path.endswith('.py'):
                        # Try to import and run hybrid system
                        return self._get_hybrid_system_data()
                    else:
                        # Try to read CSV/JSON files
                        return self._read_data_files(source_path)
                except Exception as e:
                    logger.warning(f"Could not read from {source_path}: {e}")
                    source_info['status'] = 'error'
        
        return None
    
    def _get_hybrid_system_data(self) -> Optional[Dict[str, Any]]:
        """Get data from the hybrid environmental sensing system."""
        try:
            import sys
            sys.path.append('..')
            
            # Try to import and run hybrid system
            from hybrid_environmental_sensing_system import HybridEnvironmentalSensingSystem
            
            hybrid_system = HybridEnvironmentalSensingSystem()
            results = hybrid_system.run_hybrid_analysis()
            
            if results and 'environmental_parameters' in results:
                return self._convert_hybrid_results(results['environmental_parameters'])
            
        except Exception as e:
            logger.warning(f"Could not get hybrid system data: {e}")
        
        return None
    
    def _convert_hybrid_results(self, hybrid_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert hybrid system results to dashboard format."""
        converted_data = {}
        
        for param_name, param_info in hybrid_data.items():
            if isinstance(param_info, dict) and 'value' in param_info:
                converted_data[param_name] = {
                    'value': param_info['value'],
                    'unit': param_info.get('unit', ''),
                    'timestamp': datetime.now(),
                    'confidence': param_info.get('confidence', 0.0)
                }
        
        return converted_data
    
    def _read_data_files(self, source_path: str) -> Optional[Dict[str, Any]]:
        """Read environmental data from files."""
        try:
            source_dir = Path(source_path)
            if not source_dir.is_dir():
                return None
            
            # Look for recent data files
            data_files = list(source_dir.glob('*.csv')) + list(source_dir.glob('*.json'))
            if not data_files:
                return None
            
            # Get most recent file
            latest_file = max(data_files, key=lambda f: f.stat().st_mtime)
            
            if latest_file.suffix == '.csv':
                data = pd.read_csv(latest_file)
            else:
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                    data = pd.DataFrame(data)
            
            # Extract latest values
            return self._extract_latest_values(data)
            
        except Exception as e:
            logger.warning(f"Could not read data files from {source_path}: {e}")
            return None
    
    def _extract_latest_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Extract latest values from data DataFrame."""
        latest_values = {}
        
        # Map common column names to parameters
        column_mapping = {
            'temperature': ['temp', 'temperature', 't'],
            'humidity': ['humidity', 'hum', 'h', 'rh'],
            'ph': ['ph', 'pH', 'ph_value'],
            'pollution': ['pollution', 'poll', 'contamination'],
            'moisture': ['moisture', 'moist', 'water_content']
        }
        
        for param_name, possible_columns in column_mapping.items():
            for col in possible_columns:
                if col in data.columns:
                    # Get latest non-null value
                    values = data[col].dropna()
                    if len(values) > 0:
                        latest_values[param_name] = {
                            'value': float(values.iloc[-1]),
                            'unit': self._get_parameter_unit(param_name),
                            'timestamp': datetime.now(),
                            'confidence': 0.85  # Assume good confidence for real data
                        }
                        break
        
        return latest_values
    
    def _get_parameter_unit(self, param_name: str) -> str:
        """Get the appropriate unit for a parameter."""
        units = {
            'temperature': '°C',
            'humidity': '%',
            'ph': 'pH units',
            'pollution': 'ppm',
            'moisture': '%',
            'electrical_activity': 'mV'
        }
        return units.get(param_name, '')
    
    def _generate_simulated_data(self) -> Dict[str, Any]:
        """Generate simulated environmental data for missing parameters."""
        simulated_data = {}
        
        # Base values with realistic variations
        base_values = {
            'temperature': 22.0,
            'humidity': 65.0,
            'ph': 6.8,
            'pollution': 0.05,
            'moisture': 45.0,
            'electrical_activity': 0.12
        }
        
        # Add realistic variations
        for param_name, base_value in base_values.items():
            if param_name not in self.data_buffer:
                # Generate realistic variation
                if param_name == 'temperature':
                    variation = np.random.normal(0, 1.0)
                elif param_name == 'humidity':
                    variation = np.random.normal(0, 5.0)
                elif param_name == 'ph':
                    variation = np.random.normal(0, 0.1)
                elif param_name == 'pollution':
                    variation = np.random.exponential(0.02)
                else:
                    variation = np.random.normal(0, base_value * 0.1)
                
                simulated_data[param_name] = {
                    'value': base_value + variation,
                    'unit': self._get_parameter_unit(param_name),
                    'timestamp': datetime.now(),
                    'confidence': 0.70  # Lower confidence for simulated data
                }
        
        return simulated_data
    
    def _get_fallback_data(self) -> List[EnvironmentalParameter]:
        """Get fallback data when all other sources fail."""
        fallback_params = []
        
        # Generate basic simulated parameters
        base_values = {
            'temperature': 22.0,
            'humidity': 65.0,
            'ph': 6.8,
            'pollution': 0.05,
            'moisture': 45.0
        }
        
        for param_name, value in base_values.items():
            fallback_params.append(EnvironmentalParameter(
                name=param_name,
                value=value,
                unit=self._get_parameter_unit(param_name),
                timestamp=datetime.now(),
                confidence=0.50,  # Low confidence for fallback
                source='simulated',
                status='normal'
            ))
        
        return fallback_params
    
    def _assess_parameter_status(self, param_name: str, value: float) -> str:
        """
        Assess the status of an environmental parameter.
        
        Args:
            param_name: Name of the parameter
            value: Current value
            
        Returns:
            Status string: 'normal', 'warning', 'alert', 'critical'
        """
        thresholds = self.config['alert_thresholds'].get(param_name, {})
        
        if not thresholds:
            return 'normal'
        
        # Get baseline value (could be from historical data)
        baseline = self._get_parameter_baseline(param_name)
        deviation = abs(value - baseline)
        
        if deviation >= thresholds.get('critical', float('inf')):
            return 'critical'
        elif deviation >= thresholds.get('alert', float('inf')):
            return 'alert'
        elif deviation >= thresholds.get('warning', float('inf')):
            return 'warning'
        else:
            return 'normal'
    
    def _get_parameter_baseline(self, param_name: str) -> float:
        """Get baseline value for a parameter."""
        baselines = {
            'temperature': 22.0,
            'humidity': 65.0,
            'ph': 6.8,
            'pollution': 0.0,
            'moisture': 45.0
        }
        return baselines.get(param_name, 0.0)
    
    def _update_data_buffer(self, current_params: List[EnvironmentalParameter]):
        """Update the data buffer with current parameters."""
        timestamp = datetime.now()
        
        for param in current_params:
            if param.name not in self.data_buffer:
                self.data_buffer[param.name] = []
            
            # Add new data point
            self.data_buffer[param.name].append({
                'value': param.value,
                'timestamp': timestamp,
                'status': param.status,
                'confidence': param.confidence
            })
            
            # Maintain buffer size
            if len(self.data_buffer[param.name]) > self.config['data_buffer_size']:
                self.data_buffer[param.name].pop(0)
    
    def get_system_status(self) -> SystemStatus:
        """
        Get current system health status.
        
        Returns:
            SystemStatus object with health information
        """
        try:
            # Calculate overall health
            data_quality = self._calculate_data_quality()
            sensor_status = self._get_sensor_status()
            active_alert_count = len([a for a in self.active_alerts if not a.resolved])
            
            # Overall health is weighted average of components
            overall_health = (data_quality * 0.6 + 
                            (100 - active_alert_count * 10) * 0.4)
            overall_health = max(0, min(100, overall_health))
            
            return SystemStatus(
                overall_health=overall_health,
                data_quality=data_quality,
                sensor_status=sensor_status,
                last_update=datetime.now(),
                uptime=datetime.now() - self.start_time,
                active_alerts=active_alert_count
            )
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return SystemStatus(
                overall_health=50.0,
                data_quality=50.0,
                sensor_status={},
                last_update=datetime.now(),
                uptime=datetime.now() - self.start_time,
                active_alerts=0
            )
    
    def _calculate_data_quality(self) -> float:
        """Calculate overall data quality score."""
        if not self.data_buffer:
            return 50.0
        
        quality_scores = []
        
        for param_name, data_points in self.data_buffer.items():
            if not data_points:
                continue
            
            # Calculate quality based on confidence and consistency
            avg_confidence = np.mean([dp['confidence'] for dp in data_points])
            consistency = 1.0 - np.std([dp['value'] for dp in data_points]) / 100.0
            
            param_quality = (avg_confidence * 0.7 + consistency * 0.3) * 100
            quality_scores.append(param_quality)
        
        return np.mean(quality_scores) if quality_scores else 50.0
    
    def _get_sensor_status(self) -> Dict[str, str]:
        """Get status of all data sources/sensors."""
        sensor_status = {}
        
        for source_name, source_info in self.data_sources.items():
            if source_info['status'] == 'available':
                sensor_status[source_name] = 'active'
            elif source_info['status'] == 'error':
                sensor_status[source_name] = 'error'
            else:
                sensor_status[source_name] = 'unknown'
        
        return sensor_status
    
    def start_monitoring(self):
        """Start the real-time monitoring system."""
        if self.is_running:
            logger.warning("Monitoring already running")
            return
        
        self.is_running = True
        logger.info("🚀 Starting real-time environmental monitoring")
        
        try:
            # Start monitoring loop
            asyncio.run(self._monitoring_loop())
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
        finally:
            self.is_running = False
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            start_time = time.time()
            
            try:
                # Get current parameters
                current_params = self.get_current_parameters()
                
                # Check for alerts
                new_alerts = self._check_for_alerts(current_params)
                if new_alerts:
                    self._process_alerts(new_alerts)
                
                # Update performance metrics
                self._update_performance_metrics(start_time)
                
                # Notify callbacks
                update_data = {
                    'timestamp': datetime.now(),
                    'parameters': [asdict(param) for param in current_params],
                    'system_status': asdict(self.get_system_status()),
                    'active_alerts': len(self.active_alerts)
                }
                self._notify_callbacks(update_data)
                
                # Wait for next update
                await asyncio.sleep(self.config['update_frequency'])
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1.0)  # Brief pause on error
    
    def _check_for_alerts(self, current_params: List[EnvironmentalParameter]) -> List[Alert]:
        """Check current parameters for alert conditions."""
        new_alerts = []
        
        for param in current_params:
            if param.status in ['warning', 'alert', 'critical']:
                # Check if we already have an active alert for this parameter
                existing_alert = next(
                    (a for a in self.active_alerts 
                     if a.parameter == param.name and not a.resolved),
                    None
                )
                
                if not existing_alert:
                    # Create new alert
                    alert = Alert(
                        id=f"alert_{param.name}_{int(time.time())}",
                        parameter=param.name,
                        level=param.status,
                        message=f"{param.name.replace('_', ' ').title()} is at {param.status} level: {param.value:.2f} {param.unit}",
                        timestamp=datetime.now()
                    )
                    new_alerts.append(alert)
        
        return new_alerts
    
    def _process_alerts(self, new_alerts: List[Alert]):
        """Process new alerts and add to active alerts."""
        for alert in new_alerts:
            self.active_alerts.append(alert)
            self.alert_history.append(alert)
            logger.warning(f"🚨 New alert: {alert.message}")
    
    def _update_performance_metrics(self, start_time: float):
        """Update performance tracking metrics."""
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        self.performance_metrics['total_updates'] += 1
        self.performance_metrics['last_response_time'] = response_time
        
        # Update average response time
        total = self.performance_metrics['total_updates']
        current_avg = self.performance_metrics['average_response_time']
        self.performance_metrics['average_response_time'] = (
            (current_avg * (total - 1) + response_time) / total
        )
        
        # Calculate data throughput
        self.performance_metrics['data_throughput'] = total / (
            (datetime.now() - self.start_time).total_seconds() / 3600
        )  # updates per hour
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        for alert in self.active_alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                logger.info(f"Alert acknowledged: {alert_id}")
                break
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved."""
        for alert in self.active_alerts:
            if alert.id == alert_id:
                alert.resolved = True
                logger.info(f"Alert resolved: {alert_id}")
                break
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """Get alert history."""
        return self.alert_history[-limit:] if self.alert_history else []
    
    def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts."""
        return [a for a in self.active_alerts if not a.resolved]


def main():
    """Main function for testing the Real-time Dashboard."""
    print("🧪 Testing Real-time Dashboard Backend")
    
    # Initialize dashboard
    dashboard = RealTimeDashboard()
    
    # Test basic functionality
    print("\n📊 Testing basic functionality...")
    
    # 1. Get current parameters
    current_params = dashboard.get_current_parameters()
    print(f"✅ Retrieved {len(current_params)} environmental parameters")
    
    # 2. Get system status
    system_status = dashboard.get_system_status()
    print(f"✅ System health: {system_status.overall_health:.1f}%")
    print(f"✅ Data quality: {system_status.data_quality:.1f}%")
    
    # 3. Test alert system
    print("\n🚨 Testing alert system...")
    test_alert = Alert(
        id="test_alert_001",
        parameter="temperature",
        level="warning",
        message="Test temperature warning",
        timestamp=datetime.now()
    )
    dashboard.active_alerts.append(test_alert)
    
    active_alerts = dashboard.get_active_alerts()
    print(f"✅ Active alerts: {len(active_alerts)}")
    
    # 4. Test performance metrics
    performance = dashboard.get_performance_metrics()
    print(f"✅ Performance metrics: {performance['total_updates']} updates")
    
    print("\n✅ Real-time Dashboard test completed successfully!")


if __name__ == "__main__":
    main() 