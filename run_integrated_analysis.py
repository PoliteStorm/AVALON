#!/usr/bin/env python3
"""
Integrated Analysis Runner: Adamatzky + Wave Transform

This script demonstrates the complete integrated analysis framework:
1. Adamatzky frequency discrimination analysis
2. Wave transform analysis W(k,τ) = ∫₀^∞ V(t) · ψ(√t/τ) · e^(-ik√t) dt
3. Cross-domain correlation and insights
"""

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the complete integrated analysis."""
    logger.info("🚀 Starting Integrated Adamatzky + Wave Transform Analysis")
    
    # Check if wave transform analyzer is available
    try:
        from integrated_wave_transform_analyzer import IntegratedWaveTransformAnalyzer
        logger.info("✅ Wave transform analyzer available")
        wave_transform_available = True
    except ImportError:
        logger.warning("⚠️  Wave transform analyzer not available - using basic analysis")
        wave_transform_available = False
    
    # Run enhanced Adamatzky analysis
    try:
        from enhanced_adamatzky_analysis_with_wave_transform import EnhancedAdamatzkyAnalyzer
        
        logger.info("🍄 Running Enhanced Adamatzky Analysis...")
        analyzer = EnhancedAdamatzkyAnalyzer(sampling_rate=1.0)
        
        # Load data
        data_file = "DATA/processed/validated_fungal_electrical_csvs/New_Oyster_with spray_as_mV_seconds_SigView.csv"
        
        if not Path(data_file).exists():
            logger.error(f"Data file not found: {data_file}")
            return
        
        # Load data first
        signal_data = analyzer.load_fungal_data(data_file)
        if len(signal_data) == 0:
            logger.error("No valid data loaded")
            return
        
        # Perform analysis
        results = analyzer.analyze_frequency_discrimination(signal_data)
        
        # Create visualizations
        analyzer.create_comprehensive_visualizations(results)
        
        # Save results
        analyzer.save_enhanced_results(results)
        
        logger.info("✅ Enhanced analysis completed successfully!")
        
    except ImportError:
        logger.warning("⚠️  Enhanced analyzer not available - running basic analysis")
        
        # Fall back to basic analysis
        from adamatzky_frequency_discrimination_analysis import AdamatzkyFrequencyAnalyzer
        
        analyzer = AdamatzkyFrequencyAnalyzer(sampling_rate=1.0)
        signal_data = analyzer.load_fungal_data(data_file)
        results = analyzer.analyze_frequency_discrimination(signal_data)
        analyzer.create_visualizations(results)
        analyzer.save_results(results)
        
        logger.info("✅ Basic analysis completed successfully!")
    
    logger.info("🎉 Integrated analysis framework demonstration completed!")

if __name__ == "__main__":
    main() 