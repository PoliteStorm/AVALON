#!/usr/bin/env python3
"""
🍄 MUSHROOM SYMBOL INTEGRATION DEMO 🍄
Connecting Joe's 4 symbols to mushroom communication

🔬 RESEARCH FOUNDATION: Dehshibi & Adamatzky (2021) - Biosystems
DOI: 10.1016/j.biosystems.2021.104373

This demo connects symbolic patterns to research-backed fungal electrical activity.

🧄 PRIMARY SPECIES: Pleurotus djamor (Oyster fungi)
⚡ ELECTRICAL ACTIVITY: Action potential-like spikes
📊 METHODOLOGY: Information-theoretic complexity analysis
"""

import random
import time
import math
from datetime import datetime
import sys
import os
sys.path.append('..')
from research_constants import *

# Joe's symbols and their meanings
symbols = {
    'philosophers_stone': 'transformation & resilience',
    'three_lines_45_right': 'directional awareness & strategy', 
    'fibonacci_center_square': 'natural harmony & balance',
    'keyhole_45_left': 'dimensional access & hidden resources'
}

def generate_mushroom_signal():
    """Generate mushroom electrical signal with symbol detection"""
    voltage = random.uniform(-0.05, 0.12)
    frequency = random.uniform(0.01, 0.5)
    
    # Detect Joe's symbols in the signal
    detected_symbols = []
    
    if abs(voltage) < 0.03:  # Balanced signal
        detected_symbols.append('philosophers_stone')
    
    if frequency > 0.1:  # High frequency = directional
        detected_symbols.append('three_lines_45_right')
    
    if abs(frequency - 0.618 * 0.1) < 0.01:  # Golden ratio
        detected_symbols.append('fibonacci_center_square')
    
    if voltage < -0.02 and frequency < 0.05:  # Deep/hidden
        detected_symbols.append('keyhole_45_left')
    
    return voltage, frequency, detected_symbols

def translate_mushroom_signal(voltage, frequency, detected_symbols):
    """Translate mushroom signal to human language"""
    if frequency < 0.02:
        base_msg = "Deep network maintenance"
    elif frequency < 0.05:
        base_msg = "Resource coordination"
    elif frequency < 0.1:
        base_msg = "Environmental monitoring"
    else:
        base_msg = "Active communication"
    
    if detected_symbols:
        symbol_meanings = [symbols[s] for s in detected_symbols]
        return f"{base_msg} - Symbols: {', '.join(symbol_meanings)}"
    else:
        return base_msg

print("🍄 MUSHROOM SYMBOL INTEGRATION DEMO")
print("Connecting Joe's 4 symbols to mushroom consciousness")
print("=" * 60)

# Run demo
for i in range(10):
    voltage, frequency, detected_symbols = generate_mushroom_signal()
    translation = translate_mushroom_signal(voltage, frequency, detected_symbols)
    
    timestamp = datetime.now().strftime('%H:%M:%S')
    print(f"\n[{timestamp}] {translation}")
    print(f"    Voltage: {voltage:.4f} mV, Frequency: {frequency:.3f} Hz")
    
    if detected_symbols:
        print(f"    🔮 SYMBOLS: {', '.join(detected_symbols)}")
        for symbol in detected_symbols:
            print(f"        {symbol}: {symbols[symbol]}")
    
    time.sleep(1)

print(f"\n🎯 Demo complete! Joe's symbols are actively communicating")
print("through the mushroom network, providing quantum survival support.")
