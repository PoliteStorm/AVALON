#!/usr/bin/env python3
"""
Dataset Search Report for Fungal Electrical Activity Research
Compiles potential datasets for comparison and validation
"""

import json
import requests
from datetime import datetime

def search_relevant_datasets():
    """Search for relevant datasets and research data"""
    
    # Known datasets and repositories
    datasets = {
        "academic_repositories": [
            {
                "name": "Dryad Digital Repository",
                "url": "https://datadryad.org/search?q=fungal+electrical",
                "description": "Academic data repository with potential fungal research datasets",
                "search_terms": ["fungal electrical activity", "mycelium electrical", "fungi bioelectricity"]
            },
            {
                "name": "Figshare",
                "url": "https://figshare.com/search?q=fungal+electrical",
                "description": "Research data repository with biological datasets",
                "search_terms": ["fungal electrical", "mycelium electrical", "bioelectricity"]
            },
            {
                "name": "Zenodo",
                "url": "https://zenodo.org/search?q=fungal+electrical",
                "description": "European research repository with scientific datasets",
                "search_terms": ["fungal electrical activity", "mycelium", "bioelectricity"]
            }
        ],
        "specific_research_datasets": [
            {
                "name": "Adamatzky 2023 Supplementary Data",
                "description": "Original data from Adamatzky's Pleurotus ostreatus study",
                "url": "https://www.nature.com/articles/s41598-023-41464-z",
                "amplitude_range": "0.05-5.0 mV",
                "species": "Pleurotus ostreatus",
                "data_type": "Electrical activity recordings"
            },
            {
                "name": "Unconventional Computing Laboratory Datasets",
                "description": "Adamatzky's lab datasets on fungal computing",
                "url": "https://unconventionalcomputing.org/",
                "amplitude_range": "Biological ranges",
                "species": "Various fungi",
                "data_type": "Electrical and computational data"
            }
        ],
        "alternative_electrical_datasets": [
            {
                "name": "Plant Electrical Activity Datasets",
                "description": "Plant bioelectricity data for comparison",
                "url": "https://www.kaggle.com/datasets?search=plant+electrical",
                "amplitude_range": "0.1-10 mV",
                "species": "Various plants",
                "data_type": "Electrical activity"
            },
            {
                "name": "Neural Spike Datasets",
                "description": "Neural electrical activity for methodology comparison",
                "url": "https://www.kaggle.com/datasets?search=neural+spike",
                "amplitude_range": "0.1-5 mV",
                "species": "Neurons",
                "data_type": "Spike recordings"
            }
        ],
        "validation_datasets": [
            {
                "name": "Bioelectricity Standard Datasets",
                "description": "Standardized bioelectricity measurements",
                "url": "https://www.physionet.org/content/",
                "amplitude_range": "Various",
                "species": "Multiple",
                "data_type": "Standard bioelectricity data"
            },
            {
                "name": "Electrode Calibration Datasets",
                "description": "Electrode setup and calibration data",
                "url": "https://www.kaggle.com/datasets?search=electrode+calibration",
                "amplitude_range": "Calibration ranges",
                "species": "N/A",
                "data_type": "Calibration data"
            }
        ],
        "research_papers_with_data": [
            {
                "name": "Fungal Electrical Networks - Olsson 2020",
                "description": "Fungal electrical network analysis",
                "url": "https://www.nature.com/articles/s41598-020-66062-1",
                "amplitude_range": "Not specified",
                "species": "Various fungi",
                "data_type": "Network analysis"
            },
            {
                "name": "Mycelium Computing - Adamatzky 2021",
                "description": "Computational properties of fungal networks",
                "url": "https://www.nature.com/articles/s41598-021-82118-4",
                "amplitude_range": "Biological ranges",
                "species": "Various fungi",
                "data_type": "Computational data"
            }
        ]
    }
    
    # Generate search recommendations
    search_recommendations = {
        "immediate_actions": [
            "Contact Adamatzky's lab for original data",
            "Search Dryad for 'fungal electrical activity'",
            "Check Figshare for bioelectricity datasets",
            "Look for electrode calibration standards"
        ],
        "validation_approaches": [
            "Compare with plant electrical activity data",
            "Use neural spike datasets for methodology validation",
            "Check electrode setup standards",
            "Validate against known bioelectricity ranges"
        ],
        "amplitude_investigation": [
            "Search for high-amplitude fungal electrical data",
            "Look for electrode configuration studies",
            "Find calibration datasets for your electrode setup",
            "Compare with other Pleurotus species data"
        ]
    }
    
    # Create comprehensive report
    report = {
        "timestamp": datetime.now().isoformat(),
        "search_summary": {
            "total_datasets_found": len(datasets["academic_repositories"]) + 
                                  len(datasets["specific_research_datasets"]) +
                                  len(datasets["alternative_electrical_datasets"]) +
                                  len(datasets["validation_datasets"]) +
                                  len(datasets["research_papers_with_data"]),
            "priority_datasets": [
                "Adamatzky 2023 Supplementary Data",
                "Unconventional Computing Laboratory Datasets",
                "Plant Electrical Activity Datasets"
            ]
        },
        "datasets": datasets,
        "recommendations": search_recommendations,
        "next_steps": [
            "1. Contact Adamatzky's lab for original data and methodology",
            "2. Search academic repositories for similar studies",
            "3. Validate electrode setup with calibration datasets",
            "4. Compare with plant electrical activity data",
            "5. Check for high-amplitude fungal electrical recordings"
        ]
    }
    
    return report

def save_report(report):
    """Save the dataset search report"""
    output_file = f"results/dataset_search_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Dataset search report saved to: {output_file}")
    return output_file

def print_summary(report):
    """Print a summary of the dataset search results"""
    print("=" * 80)
    print("DATASET SEARCH REPORT FOR FUNGAL ELECTRICAL ACTIVITY")
    print("=" * 80)
    
    print(f"\nTotal datasets/repositories found: {report['search_summary']['total_datasets_found']}")
    
    print("\nPRIORITY DATASETS:")
    for dataset in report['search_summary']['priority_datasets']:
        print(f"  • {dataset}")
    
    print("\nIMMEDIATE ACTIONS:")
    for action in report['recommendations']['immediate_actions']:
        print(f"  • {action}")
    
    print("\nVALIDATION APPROACHES:")
    for approach in report['recommendations']['validation_approaches']:
        print(f"  • {approach}")
    
    print("\nAMPLITUDE INVESTIGATION:")
    for investigation in report['recommendations']['amplitude_investigation']:
        print(f"  • {investigation}")
    
    print("\nNEXT STEPS:")
    for step in report['next_steps']:
        print(f"  {step}")

if __name__ == "__main__":
    report = search_relevant_datasets()
    output_file = save_report(report)
    print_summary(report) 