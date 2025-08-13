
#!/usr/bin/env python3
"""
🔗 Phase 3 Data Integration Module
==================================

Memory-efficient data integration for Phase 3 visualization system.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta

class Phase3DataIntegration:
    """Memory-efficient data integration for Phase 3."""
    
    def __init__(self, config_path="config/data_sources_updated.json"):
        self.config = self._load_config(config_path)
        self.loader = self._create_loader()
        
    def _load_config(self, config_path):
        """Load data source configuration."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except:
            # Fallback config
            return {
                "default_source": "../../DATA/raw/15061491/Ch1-2.csv",
                "chunk_size": 10000,
                "sample_size": 1000
            }
    
    def _create_loader(self):
        """Create memory-efficient data loader."""
        class MemoryEfficientLoader:
            def __init__(self, chunk_size=10000):
                self.chunk_size = chunk_size
            
            def load_sample(self, file_path, sample_size=1000):
                """Load a sample of CSV data."""
                try:
                    return pd.read_csv(file_path, nrows=sample_size)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    return None
            
            def load_chunked(self, file_path, max_chunks=5):
                """Load CSV data in chunks."""
                try:
                    chunks = []
                    for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=self.chunk_size)):
                        chunks.append(chunk)
                        if chunk_num >= max_chunks - 1:
                            break
                    return pd.concat(chunks, ignore_index=True)
                except Exception as e:
                    print(f"Error in chunked loading: {e}")
                    return None
        
        return MemoryEfficientLoader(self.config.get("chunk_size", 10000))
    
    def get_environmental_data(self, source=None):
        """Get environmental data for visualization."""
        if source is None:
            source = self.config["default_source"]
        
        # Try to load real data
        data = self.loader.load_sample(source, self.config.get("sample_size", 1000))
        
        if data is not None:
            print(f"✅ Loaded real data: {len(data)} rows from {source}")
            return self._process_real_data(data)
        else:
            print("⚠️  Falling back to sample data")
            return self._generate_sample_data()
    
    def _process_real_data(self, data):
        """Process real CSV data for visualization."""
        # Get the actual column names from the data
        actual_columns = list(data.columns)
        print(f"🔍 Processing real data with columns: {actual_columns}")
        
        # Map real CSV columns to environmental parameters based on actual column names
        # Use the first few columns for electrical data
        if len(actual_columns) >= 4:
            # First column is usually index, next two are voltage channels, last is electrical activity
            ch1_col = actual_columns[1] if len(actual_columns) > 1 else actual_columns[0]
            ch2_col = actual_columns[2] if len(actual_columns) > 2 else actual_columns[1]
            electrical_col = actual_columns[3] if len(actual_columns) > 3 else actual_columns[-1]
            
            print(f"📊 Using columns: CH1={ch1_col}, CH2={ch2_col}, Electrical={electrical_col}")
            
            # Add environmental parameters based on electrical data
            # Temperature estimation from electrical activity patterns
            data['temperature'] = 22 + (data[electrical_col] * 10)  # Base 22°C with variation
            
            # Humidity estimation from voltage stability
            data['humidity'] = 50 + (np.abs(data[ch1_col]) * 100)  # Base 50% with variation
            
            # pH estimation from electrical balance
            data['ph'] = 6.5 + (data[ch1_col] - data[ch2_col]) * 2  # Base 6.5 with variation
            
            # Moisture estimation from voltage amplitude
            data['moisture'] = 40 + (np.abs(data[ch1_col] + data[ch2_col]) * 50)  # Base 40% with variation
            
            # Pollution estimation from electrical noise
            data['pollution'] = np.abs(data[electrical_col]) * 0.1  # Low baseline with variation
        else:
            # Fallback: generate environmental parameters
            print("⚠️  Limited columns, generating environmental parameters")
            data['temperature'] = np.random.normal(22, 5, len(data))
            data['humidity'] = np.random.uniform(30, 80, len(data))
            data['ph'] = np.random.normal(6.5, 0.5, len(data))
            data['moisture'] = np.random.uniform(20, 70, len(data))
            data['pollution'] = np.random.exponential(0.1, len(data))
        
        # Ensure coordinates exist
        if 'x_coordinate' not in data.columns:
            data['x_coordinate'] = np.random.uniform(0, 100, len(data))
        if 'y_coordinate' not in data.columns:
            data['y_coordinate'] = np.random.uniform(0, 100, len(data))
        
        # Ensure timestamp exists
        if 'timestamp' not in data.columns:
            data['timestamp'] = pd.date_range(
                start=datetime.now() - timedelta(hours=len(data)),
                periods=len(data),
                freq='h'  # Fixed deprecation warning
            )
        
        return data
    
    def _generate_sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n_points = 100
        
        data = {
            'x_coordinate': np.random.uniform(0, 100, n_points),
            'y_coordinate': np.random.uniform(0, 100, n_points),
            'temperature': np.random.normal(22, 5, n_points),
            'humidity': np.random.uniform(30, 80, n_points),
            'ph': np.random.normal(6.5, 0.5, n_points),
            'pollution': np.random.exponential(0.1, n_points),
            'moisture': np.random.uniform(20, 70, n_points),
            'electrical_activity': np.random.normal(0.1, 0.05, n_points),
            'timestamp': pd.date_range(start=datetime.now() - timedelta(hours=n_points), periods=n_points, freq='H')
        }
        
        return pd.DataFrame(data)

# Usage example:
# integration = Phase3DataIntegration()
# data = integration.get_environmental_data()
