import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

class EnhancedFungalAcousticProtocol:
    def __init__(self):
        # Simulation results show these are the key challenges
        self.challenges = {
            'ultra_low_frequency': 'Frequencies 0.00007-0.0008 Hz are below standard audio range',
            'small_pressures': 'Pressures 1e-6 to 1e-4 Pa need specialized sensors',
            'long_intervals': '41-116 minute intervals require extended monitoring',
            'environmental_noise': 'Thermal and electrical noise masking signals'
        }
        
        # Joe's species data from simulation
        self.species_priorities = {
            'S. commune': {
                'priority': 1,
                'reason': 'Most active (35 spikes), shortest intervals (41 min)',
                'sensitivity_needed': '4.40e-07 Pa',
                'frequency_range': '2.03e-04 - 8.13e-04 Hz'
            },
            'F. velutipes': {
                'priority': 2,
                'reason': 'Highest voltage (0.3 mV), good activity (14 spikes)',
                'sensitivity_needed': '1.03e-04 Pa',
                'frequency_range': '8.17e-05 - 3.27e-04 Hz'
            },
            'C. militaris': {
                'priority': 3,
                'reason': 'Moderate activity (12 spikes), 116 min intervals',
                'sensitivity_needed': '1.07e-04 Pa',
                'frequency_range': '7.18e-05 - 2.87e-04 Hz'
            },
            'O. nidiformis': {
                'priority': 4,
                'reason': 'Lowest voltage (0.007 mV), challenging detection',
                'sensitivity_needed': '1.56e-06 Pa',
                'frequency_range': '9.06e-05 - 3.62e-04 Hz'
            }
        }
    
    def enhanced_equipment_recommendations(self):
        """
        Specific equipment recommendations based on simulation results
        """
        equipment = {
            'primary_sensors': {
                'hydrophone': {
                    'model': 'Brüel & Kjær 8103 or equivalent',
                    'sensitivity': '1e-7 Pa minimum',
                    'frequency_range': '0.1 Hz - 180 kHz',
                    'cost': '$3,000-5,000',
                    'why': 'Ultra-low frequency detection, high sensitivity'
                },
                'accelerometer': {
                    'model': 'PCB Piezotronics 393B04 or equivalent',
                    'sensitivity': '100 mV/g',
                    'frequency_range': '0.15 Hz - 1 kHz',
                    'cost': '$500-800',
                    'why': 'Detect substrate vibrations from acoustic waves'
                },
                'infrasound_microphone': {
                    'model': 'Chaparral Physics Model 25 or equivalent',
                    'sensitivity': '1e-6 Pa',
                    'frequency_range': '0.01 Hz - 500 Hz',
                    'cost': '$8,000-12,000',
                    'why': 'Specialized for ultra-low frequency detection'
                }
            },
            'signal_processing': {
                'high_resolution_adc': {
                    'model': 'National Instruments USB-4431 or equivalent',
                    'resolution': '24-bit',
                    'sample_rate': '102.4 kS/s',
                    'cost': '$2,000-3,000',
                    'why': 'Capture ultra-low frequency signals accurately'
                },
                'low_noise_preamplifier': {
                    'model': 'Stanford Research SR560 or equivalent',
                    'gain': '1 to 50,000',
                    'noise': '4 nV/√Hz',
                    'cost': '$2,500-4,000',
                    'why': 'Amplify weak signals without adding noise'
                }
            },
            'environmental_control': {
                'acoustic_isolation_chamber': {
                    'specifications': 'Double-wall construction, acoustic foam',
                    'vibration_isolation': 'Pneumatic isolation table',
                    'cost': '$5,000-10,000',
                    'why': 'Eliminate environmental noise'
                },
                'temperature_control': {
                    'stability': '±0.1°C',
                    'range': '20-30°C',
                    'cost': '$1,000-2,000',
                    'why': 'Maintain consistent fungal activity'
                }
            }
        }
        
        return equipment
    
    def optimized_experimental_protocol(self):
        """
        Step-by-step experimental protocol optimized for success
        """
        protocol = {
            'phase_1_proof_of_concept': {
                'duration': '2-4 weeks',
                'budget': '$15,000-25,000',
                'equipment': 'Hydrophone + accelerometer + basic isolation',
                'species': 'S. commune (highest activity)',
                'goals': [
                    'Detect any acoustic signatures',
                    'Correlate with electrical activity',
                    'Establish baseline noise levels',
                    'Optimize sensor placement'
                ],
                'success_criteria': 'Signal-to-noise ratio > 3:1 at expected frequencies'
            },
            'phase_2_full_characterization': {
                'duration': '2-3 months',
                'budget': '$30,000-50,000',
                'equipment': 'Full sensor array + advanced processing',
                'species': 'All four species',
                'goals': [
                    'Characterize acoustic signatures for all species',
                    'Build acoustic-linguistic database',
                    'Validate temporal patterns',
                    'Develop translation algorithms'
                ],
                'success_criteria': 'Reproducible acoustic patterns correlated with electrical activity'
            },
            'phase_3_rosetta_stone': {
                'duration': '6-12 months',
                'budget': '$50,000-100,000',
                'equipment': 'Research-grade facility',
                'species': 'Extended studies with environmental variables',
                'goals': [
                    'Complete acoustic-linguistic mapping',
                    'Develop fungal communication translator',
                    'Test across different environmental conditions',
                    'Prepare for peer review publication'
                ],
                'success_criteria': 'Functional fungal communication translation system'
            }
        }
        
        return protocol
    
    def signal_enhancement_techniques(self):
        """
        Advanced signal processing techniques to improve detection
        """
        techniques = {
            'frequency_domain_analysis': {
                'method': 'FFT with ultra-long windows',
                'parameters': 'Window length: 24-48 hours',
                'benefit': 'Resolve ultra-low frequency components',
                'implementation': 'MATLAB or Python with scipy'
            },
            'adaptive_filtering': {
                'method': 'Wiener filtering with noise estimation',
                'parameters': 'Update filter every 1 hour',
                'benefit': 'Remove environmental noise adaptively',
                'implementation': 'Real-time DSP algorithms'
            },
            'correlation_analysis': {
                'method': 'Cross-correlation with electrical signals',
                'parameters': 'Time lag: ±10 seconds',
                'benefit': 'Identify acoustic events correlated with electrical activity',
                'implementation': 'Custom correlation software'
            },
            'template_matching': {
                'method': 'Matched filter with expected waveforms',
                'parameters': 'Template based on piezoelectric model',
                'benefit': 'Detect specific acoustic signatures',
                'implementation': 'Template-based detection algorithms'
            }
        }
        
        return techniques
    
    def alternative_detection_methods(self):
        """
        Alternative methods if direct acoustic detection fails
        """
        alternatives = {
            'substrate_vibration_analysis': {
                'method': 'Accelerometer array on growth medium',
                'sensitivity': '10x higher than air-coupled acoustic',
                'equipment': 'Multiple accelerometers, synchronization',
                'cost': '$5,000-8,000',
                'success_probability': '60-70%'
            },
            'optical_interferometry': {
                'method': 'Laser interferometer detecting surface displacement',
                'sensitivity': 'Nanometer-scale movements',
                'equipment': 'Laser interferometer system',
                'cost': '$50,000-100,000',
                'success_probability': '80-90%'
            },
            'electrical_field_mapping': {
                'method': 'High-resolution electrical field measurements',
                'sensitivity': 'Microvolt precision',
                'equipment': 'Electrode arrays, low-noise amplifiers',
                'cost': '$10,000-15,000',
                'success_probability': '70-80%'
            },
            'electromagnetic_emission': {
                'method': 'Detect EM radiation from electrical activity',
                'sensitivity': 'RF spectrum analysis',
                'equipment': 'Spectrum analyzer, RF antennas',
                'cost': '$8,000-12,000',
                'success_probability': '50-60%'
            }
        }
        
        return alternatives
    
    def create_realistic_timeline(self):
        """
        Create realistic timeline for Joe's research
        """
        timeline = {
            'Month 1-2': {
                'activities': [
                    'Acquire basic acoustic detection equipment',
                    'Set up S. commune cultures',
                    'Establish electrical monitoring baseline',
                    'Initial acoustic measurements'
                ],
                'budget': '$10,000-15,000',
                'expected_outcomes': 'Proof of concept data'
            },
            'Month 3-4': {
                'activities': [
                    'Optimize sensor placement and isolation',
                    'Implement signal processing algorithms',
                    'Extend monitoring to other species',
                    'Correlate acoustic with electrical data'
                ],
                'budget': '$15,000-20,000',
                'expected_outcomes': 'First acoustic-electrical correlations'
            },
            'Month 5-8': {
                'activities': [
                    'Full characterization of all species',
                    'Develop acoustic-linguistic database',
                    'Test environmental variables',
                    'Refine detection algorithms'
                ],
                'budget': '$20,000-30,000',
                'expected_outcomes': 'Complete acoustic signatures database'
            },
            'Month 9-12': {
                'activities': [
                    'Develop translation algorithms',
                    'Test across different conditions',
                    'Prepare research papers',
                    'Seek additional funding/collaborations'
                ],
                'budget': '$25,000-40,000',
                'expected_outcomes': 'Functional fungal communication translator'
            }
        }
        
        return timeline
    
    def generate_comprehensive_report(self):
        """
        Generate comprehensive experimental plan
        """
        print("🚀 Enhanced Fungal Acoustic Detection Protocol")
        print("=" * 60)
        print("Based on realistic simulation results")
        print()
        
        # Equipment recommendations
        print("1. EQUIPMENT RECOMMENDATIONS:")
        equipment = self.enhanced_equipment_recommendations()
        
        print("\n   Primary Sensors:")
        for sensor, details in equipment['primary_sensors'].items():
            print(f"     {sensor.replace('_', ' ').title()}: {details['model']}")
            print(f"       Sensitivity: {details['sensitivity']}")
            print(f"       Cost: {details['cost']}")
            print(f"       Why: {details['why']}")
        
        # Species priority
        print("\n2. SPECIES TESTING PRIORITY:")
        for species, info in self.species_priorities.items():
            print(f"     {info['priority']}. {species}: {info['reason']}")
            print(f"        Required sensitivity: {info['sensitivity_needed']}")
        
        # Experimental protocol
        print("\n3. EXPERIMENTAL PROTOCOL:")
        protocol = self.optimized_experimental_protocol()
        
        for phase, details in protocol.items():
            print(f"\n   {phase.replace('_', ' ').title()}:")
            print(f"     Duration: {details['duration']}")
            print(f"     Budget: {details['budget']}")
            print(f"     Goals: {details['goals'][0]} (+ {len(details['goals'])-1} more)")
        
        # Signal enhancement
        print("\n4. SIGNAL ENHANCEMENT TECHNIQUES:")
        techniques = self.signal_enhancement_techniques()
        
        for tech, details in techniques.items():
            print(f"     {tech.replace('_', ' ').title()}: {details['method']}")
            print(f"       Benefit: {details['benefit']}")
        
        # Alternative methods
        print("\n5. ALTERNATIVE DETECTION METHODS:")
        alternatives = self.alternative_detection_methods()
        
        for method, details in alternatives.items():
            print(f"     {method.replace('_', ' ').title()}: {details['success_probability']} success rate")
            print(f"       Cost: {details['cost']}")
        
        # Timeline
        print("\n6. REALISTIC TIMELINE:")
        timeline = self.create_realistic_timeline()
        
        for period, activities in timeline.items():
            print(f"     {period}: {activities['expected_outcomes']}")
            print(f"       Budget: {activities['budget']}")
        
        # Save comprehensive report
        report_data = {
            'equipment_recommendations': equipment,
            'species_priorities': self.species_priorities,
            'experimental_protocol': protocol,
            'signal_enhancement': techniques,
            'alternative_methods': alternatives,
            'timeline': timeline
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'enhanced_fungal_acoustic_protocol_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n✅ Complete protocol saved to {filename}")
        
        # Key recommendations
        print("\n" + "="*60)
        print("KEY RECOMMENDATIONS FOR JOE:")
        print("="*60)
        print("1. START WITH S. COMMUNE - highest activity, best chance of success")
        print("2. INVEST IN HYDROPHONE - specialized for ultra-low frequencies")
        print("3. VIBRATION ISOLATION - critical for eliminating noise")
        print("4. LONG MONITORING PERIODS - 24-48 hours minimum")
        print("5. CORRELATE WITH ELECTRICAL - validate acoustic signatures")
        print("6. CONSIDER ALTERNATIVES - substrate vibration if acoustic fails")
        print("7. BUDGET: $15K-25K for proof of concept")
        print("8. TIMELINE: 2-4 weeks for initial results")
        
        return report_data

if __name__ == "__main__":
    protocol = EnhancedFungalAcousticProtocol()
    report = protocol.generate_comprehensive_report() 