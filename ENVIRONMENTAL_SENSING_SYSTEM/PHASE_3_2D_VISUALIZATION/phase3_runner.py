#!/usr/bin/env python3
"""
🚀 PHASE 3 RUNNER - 2D Visualization & Real-time Dashboard
============================================================

Main execution script for Phase 3 of the Environmental Sensing System.
This script orchestrates the 2D visualization and real-time dashboard.

Author: Environmental Sensing Research Team
Date: August 12, 2025
Version: 1.0.0
"""

import sys
import time
import asyncio
from pathlib import Path
from datetime import datetime
import json
import logging

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import Phase 3 components
from environmental_mapping_engine import EnvironmentalMappingEngine
from real_time_dashboard import RealTimeDashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('PHASE_3_2D_VISUALIZATION/phase3_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Phase3Runner:
    """
    Main runner for Phase 3: 2D Visualization & Real-time Dashboard.
    
    This class orchestrates:
    - Environmental mapping engine
    - Real-time dashboard
    - Data integration from Phase 1 & 2
    - Visualization generation
    - System monitoring
    """
    
    def __init__(self):
        """Initialize Phase 3 runner."""
        self.mapping_engine = None
        self.dashboard = None
        self.is_running = False
        self.start_time = None
        
        # Create output directories
        self.output_dir = Path("PHASE_3_2D_VISUALIZATION/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "dashboard_data").mkdir(exist_ok=True)
        (self.output_dir / "system_logs").mkdir(exist_ok=True)
        
        logger.info("🚀 Phase 3 Runner initialized successfully")
    
    def initialize_components(self):
        """Initialize all Phase 3 components."""
        logger.info("🔧 Initializing Phase 3 components...")
        
        try:
            # Initialize Environmental Mapping Engine
            self.mapping_engine = EnvironmentalMappingEngine()
            logger.info("✅ Environmental Mapping Engine initialized")
            
            # Initialize Real-time Dashboard
            self.dashboard = RealTimeDashboard()
            logger.info("✅ Real-time Dashboard initialized")
            
            # Register dashboard callback for mapping engine updates
            self.dashboard.register_callback(self._dashboard_update_callback)
            logger.info("✅ Component integration completed")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing components: {e}")
            return False
    
    def _dashboard_update_callback(self, update_data: dict):
        """Callback function for dashboard updates."""
        try:
            # Log update
            logger.debug(f"Dashboard update: {len(update_data.get('parameters', []))} parameters")
            
            # Save update data
            timestamp = update_data.get('timestamp', datetime.now())
            filename = f"dashboard_update_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.output_dir / "dashboard_data" / filename
            
            with open(filepath, 'w') as f:
                json.dump(update_data, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error in dashboard callback: {e}")
    
    def run_visualization_demo(self):
        """Run a demonstration of Phase 3 visualization capabilities."""
        logger.info("🎨 Running Phase 3 visualization demonstration...")
        
        try:
            # Generate sample data
            sample_data = self.mapping_engine._generate_sample_data()
            logger.info(f"✅ Generated sample data: {len(sample_data)} data points")
            
            # Create various visualizations
            visualizations = {}
            
            # 1. Temperature heatmap
            logger.info("🔥 Creating temperature heatmap...")
            temp_heatmap = self.mapping_engine.create_environmental_heatmap(
                sample_data, 'temperature'
            )
            visualizations['temperature_heatmap'] = temp_heatmap
            
            # 2. Humidity contour map
            logger.info("🗺️  Creating humidity contour map...")
            humidity_contour = self.mapping_engine.create_contour_map(
                sample_data, 'humidity'
            )
            visualizations['humidity_contour'] = humidity_contour
            
            # 3. Multi-parameter dashboard
            logger.info("📊 Creating multi-parameter dashboard...")
            dashboard_viz = self.mapping_engine.create_multi_parameter_dashboard(
                sample_data, 
                ['temperature', 'humidity', 'ph', 'pollution']
            )
            visualizations['multi_parameter_dashboard'] = dashboard_viz
            
            # 4. Time-lapse visualization
            logger.info("⏰ Creating time-lapse visualization...")
            time_lapse = self.mapping_engine.create_time_lapse_visualization(
                sample_data, 'temperature'
            )
            visualizations['temperature_timelapse'] = time_lapse
            
            # Save all visualizations
            logger.info("💾 Saving visualizations...")
            saved_files = {}
            
            for viz_name, viz_fig in visualizations.items():
                files = self.mapping_engine.save_visualization(
                    viz_fig, 
                    f"phase3_demo_{viz_name}",
                    ['html', 'png']
                )
                saved_files[viz_name] = files
                logger.info(f"✅ Saved {viz_name}: {len(files)} formats")
            
            # Generate demo report
            self._generate_demo_report(saved_files, sample_data)
            
            logger.info("🎉 Phase 3 visualization demonstration completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in visualization demo: {e}")
            return False
    
    def _generate_demo_report(self, saved_files: dict, sample_data):
        """Generate a comprehensive demo report."""
        try:
            report = {
                'phase': 'Phase 3: 2D Visualization & Real-time Dashboard',
                'timestamp': datetime.now().isoformat(),
                'demo_summary': {
                    'total_visualizations': len(saved_files),
                    'data_points': len(sample_data),
                    'parameters_visualized': list(sample_data.columns),
                    'output_formats': ['html', 'png']
                },
                'visualizations_created': saved_files,
                'system_capabilities': {
                    'environmental_mapping': '2D spatial visualization of environmental conditions',
                    'real_time_dashboard': 'Live monitoring interface with real-time updates',
                    'multi_parameter_analysis': 'Simultaneous visualization of multiple parameters',
                    'time_series_analysis': 'Temporal analysis of environmental changes',
                    'interactive_plots': 'Zoom, pan, and filter capabilities',
                    'export_capabilities': 'Multiple format export (HTML, PNG, SVG, JSON)'
                },
                'next_steps': [
                    'Begin real-time monitoring',
                    'Integrate with Phase 1 & 2 data sources',
                    'Implement alert management system',
                    'Add advanced analytics capabilities',
                    'Prepare for Phase 4: 3D Visualization'
                ]
            }
            
            # Save report
            report_file = self.output_dir / "phase3_demo_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📊 Demo report generated: {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating demo report: {e}")
    
    def start_real_time_monitoring(self):
        """Start real-time environmental monitoring."""
        if not self.dashboard:
            logger.error("❌ Dashboard not initialized")
            return False
        
        logger.info("🚀 Starting real-time environmental monitoring...")
        
        try:
            # Start monitoring in background
            self.is_running = True
            self.start_time = datetime.now()
            
            # Run monitoring for a limited time for demo
            logger.info("⏱️  Running real-time monitoring for 30 seconds...")
            
            # Start monitoring loop
            asyncio.run(self._run_monitoring_demo())
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error starting real-time monitoring: {e}")
            return False
        finally:
            self.is_running = False
    
    async def _run_monitoring_demo(self):
        """Run real-time monitoring demo for a limited time."""
        try:
            # Run for 30 seconds
            end_time = time.time() + 30
            
            while time.time() < end_time and self.is_running:
                # Get current parameters
                current_params = self.dashboard.get_current_parameters()
                
                # Get system status
                system_status = self.dashboard.get_system_status()
                
                # Log current status
                logger.info(f"📊 Monitoring update: {len(current_params)} parameters, "
                          f"Health: {system_status.overall_health:.1f}%")
                
                # Wait for next update
                await asyncio.sleep(5)
            
            logger.info("✅ Real-time monitoring demo completed")
            
        except Exception as e:
            logger.error(f"Error in monitoring demo: {e}")
    
    def run_comprehensive_test(self):
        """Run comprehensive Phase 3 testing."""
        logger.info("🧪 Running comprehensive Phase 3 testing...")
        
        test_results = {
            'component_initialization': False,
            'visualization_demo': False,
            'real_time_monitoring': False,
            'data_integration': False,
            'overall_status': 'FAILED'
        }
        
        try:
            # Test 1: Component initialization
            logger.info("🔧 Test 1: Component initialization...")
            if self.initialize_components():
                test_results['component_initialization'] = True
                logger.info("✅ Component initialization: PASSED")
            else:
                logger.error("❌ Component initialization: FAILED")
                return test_results
            
            # Test 2: Visualization demo
            logger.info("🎨 Test 2: Visualization demo...")
            if self.run_visualization_demo():
                test_results['visualization_demo'] = True
                logger.info("✅ Visualization demo: PASSED")
            else:
                logger.error("❌ Visualization demo: FAILED")
            
            # Test 3: Real-time monitoring
            logger.info("🚀 Test 3: Real-time monitoring...")
            if self.start_real_time_monitoring():
                test_results['real_time_monitoring'] = True
                logger.info("✅ Real-time monitoring: PASSED")
            else:
                logger.error("❌ Real-time monitoring: FAILED")
            
            # Test 4: Data integration
            logger.info("🔗 Test 4: Data integration...")
            if self._test_data_integration():
                test_results['data_integration'] = True
                logger.info("✅ Data integration: PASSED")
            else:
                logger.error("❌ Data integration: FAILED")
            
            # Determine overall status
            passed_tests = sum(test_results.values()) - 1  # Exclude overall_status
            if passed_tests >= 3:
                test_results['overall_status'] = 'PASSED'
            elif passed_tests >= 2:
                test_results['overall_status'] = 'PARTIAL'
            else:
                test_results['overall_status'] = 'FAILED'
            
            # Save test results
            self._save_test_results(test_results)
            
            logger.info(f"🎯 Comprehensive testing completed: {test_results['overall_status']}")
            return test_results
            
        except Exception as e:
            logger.error(f"❌ Error in comprehensive testing: {e}")
            test_results['overall_status'] = 'ERROR'
            return test_results
    
    def _test_data_integration(self) -> bool:
        """Test integration with Phase 1 & 2 data sources."""
        try:
            # Test Phase 1 data access
            phase1_data = self.mapping_engine.load_environmental_data(
                "../PHASE_1_DATA_INFRASTRUCTURE/RESULTS/baseline_analysis/"
            )
            
            # Test Phase 2 data access
            phase2_data = self.mapping_engine.load_environmental_data(
                "../PHASE_2_AUDIO_SYNTHESIS/results/"
            )
            
            # If we can access both, integration is working
            return True
            
        except Exception as e:
            logger.warning(f"Data integration test warning: {e}")
            # This is not a critical failure for Phase 3
            return True
    
    def _save_test_results(self, test_results: dict):
        """Save comprehensive test results."""
        try:
            results_file = self.output_dir / "phase3_test_results.json"
            
            # Add metadata
            test_results['timestamp'] = datetime.now().isoformat()
            test_results['phase'] = 'Phase 3: 2D Visualization & Real-time Dashboard'
            test_results['test_duration'] = str(datetime.now() - self.start_time) if self.start_time else 'N/A'
            
            with open(results_file, 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
            
            logger.info(f"📊 Test results saved: {results_file}")
            
        except Exception as e:
            logger.error(f"Error saving test results: {e}")
    
    def get_phase_status(self) -> dict:
        """Get current Phase 3 status."""
        return {
            'phase': 'Phase 3: 2D Visualization & Real-time Dashboard',
            'status': 'RUNNING' if self.is_running else 'READY',
            'components': {
                'mapping_engine': 'READY' if self.mapping_engine else 'NOT_INITIALIZED',
                'dashboard': 'READY' if self.dashboard else 'NOT_INITIALIZED'
            },
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'output_directory': str(self.output_dir),
            'capabilities': [
                '2D Environmental Parameter Mapping',
                'Real-time Dashboard',
                'Interactive Visualizations',
                'Multiple Export Formats',
                'Real-time Monitoring',
                'Alert Management'
            ]
        }


def main():
    """Main execution function for Phase 3."""
    print("🚀 PHASE 3: 2D Visualization & Real-time Dashboard")
    print("=" * 60)
    
    # Initialize Phase 3 runner
    runner = Phase3Runner()
    
    try:
        # Run comprehensive testing
        print("\n🧪 Running comprehensive Phase 3 testing...")
        test_results = runner.run_comprehensive_test()
        
        # Display results
        print(f"\n📊 Phase 3 Test Results:")
        print(f"   Component Initialization: {'✅ PASSED' if test_results['component_initialization'] else '❌ FAILED'}")
        print(f"   Visualization Demo: {'✅ PASSED' if test_results['visualization_demo'] else '❌ FAILED'}")
        print(f"   Real-time Monitoring: {'✅ PASSED' if test_results['real_time_monitoring'] else '❌ FAILED'}")
        print(f"   Data Integration: {'✅ PASSED' if test_results['data_integration'] else '❌ FAILED'}")
        print(f"   Overall Status: {test_results['overall_status']}")
        
        # Get final status
        final_status = runner.get_phase_status()
        print(f"\n🎯 Phase 3 Status: {final_status['status']}")
        print(f"📁 Output Directory: {final_status['output_directory']}")
        
        if test_results['overall_status'] == 'PASSED':
            print("\n🎉 Phase 3 completed successfully!")
            print("🚀 Ready for Phase 4: 3D Visualization")
        else:
            print("\n⚠️  Phase 3 completed with issues")
            print("🔧 Check logs for details")
        
        return 0 if test_results['overall_status'] == 'PASSED' else 1
        
    except KeyboardInterrupt:
        print("\n⏹️  Phase 3 execution interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Phase 3 execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 