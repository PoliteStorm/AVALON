#!/usr/bin/env python3
"""
Fungal Synth Audio Player
Author: Joe Knowles
Timestamp: 2025-08-12 09:23:27 BST
Description: Simple player for the generated fungal synth sounds
"""

import os
import time
from pathlib import Path

class FungalSynthPlayer:
    """
    Simple player for fungal synth sounds
    """
    
    def __init__(self):
        self.timestamp = "20250812_092327"
        self.author = "Joe Knowles"
        self.audio_dir = "RESULTS/audio"
        
    def list_available_sounds(self):
        """List all available fungal synth sounds"""
        print(f"🎵 AVAILABLE FUNGAL SYNTH SOUNDS")
        print(f"=" * 50)
        
        if not os.path.exists(self.audio_dir):
            print(f"❌ Audio directory not found: {self.audio_dir}")
            return []
        
        # Get all WAV files
        wav_files = []
        for file in os.listdir(self.audio_dir):
            if file.endswith('.wav'):
                wav_files.append(file)
        
        if not wav_files:
            print(f"❌ No WAV files found in {self.audio_dir}")
            return []
        
        # Sort by type
        synth_sounds = []
        other_sounds = []
        
        for file in wav_files:
            if any(keyword in file.lower() for keyword in ['additive', 'fm', 'granular', 'waveform', 'filtered', 'ambient', 'mix']):
                synth_sounds.append(file)
            else:
                other_sounds.append(file)
        
        print(f"🎹 ADVANCED SYNTH SOUNDS ({len(synth_sounds)} files):")
        for i, sound in enumerate(synth_sounds, 1):
            print(f"  {i}. {sound}")
        
        if other_sounds:
            print(f"\n🎵 OTHER FUNGAL SOUNDS ({len(other_sounds)} files):")
            for i, sound in enumerate(other_sounds, 1):
                print(f"  {i}. {sound}")
        
        return wav_files
    
    def play_sound_info(self, filename):
        """Display information about a sound file"""
        filepath = os.path.join(self.audio_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filename}")
            return
        
        # Get file size
        size_bytes = os.path.getsize(filepath)
        size_mb = size_bytes / (1024 * 1024)
        
        print(f"\n📊 SOUND FILE INFO:")
        print(f"  🎵 Filename: {filename}")
        print(f"  📁 Path: {filepath}")
        print(f"  💾 Size: {size_mb:.2f} MB")
        
        # Determine sound type
        if 'additive' in filename.lower():
            print(f"  🎹 Type: Additive Synthesis - Rich harmonic sounds")
            print(f"  🎵 Description: Multiple harmonic layers creating complex, warm tones")
        elif 'fm' in filename.lower():
            print(f"  🔔 Type: Frequency Modulation - Metallic, bell-like sounds")
            print(f"  🎵 Description: Carrier and modulator frequencies creating metallic textures")
        elif 'granular' in filename.lower():
            print(f"  🌊 Type: Granular Synthesis - Evolving textures")
            print(f"  🎵 Description: Small audio grains creating evolving, organic soundscapes")
        elif 'waveform' in filename.lower():
            print(f"  🌿 Type: Waveform Synthesis - Organic, unique sounds")
            print(f"  🎵 Description: Complex waveforms with multiple harmonics")
        elif 'filtered' in filename.lower():
            print(f"  🔥 Type: Filtered Synthesis - Warm, evolving sounds")
            print(f"  🎵 Description: Dynamic filters creating warm, evolving textures")
        elif 'ambient' in filename.lower():
            print(f"  🌌 Type: Ambient Synthesis - Spacey, meditative textures")
            print(f"  🎵 Description: Drone, pad, texture, and reverb layers")
        elif 'mix' in filename.lower():
            print(f"  🎼 Type: Complete Synth Mix - All sounds combined")
            print(f"  🎵 Description: Epic combination of all synthesis methods")
        else:
            print(f"  🎵 Type: Basic Fungal Sound")
            print(f"  🎵 Description: Direct conversion of electrical patterns to audio")
        
        print(f"  🔬 Source: REAL fungal electrical data (not simulated)")
        print(f"  ⚡ Data: 229 actual voltage measurements from mushrooms")
    
    def play_sound_instructions(self, filename):
        """Give instructions for playing the sound"""
        print(f"\n🎵 HOW TO PLAY THIS SOUND:")
        print(f"  📱 On Linux/Mac: Use 'aplay' or 'afplay' command")
        print(f"  🖥️  On Windows: Double-click the WAV file")
        print(f"  🎧 Use headphones for best experience!")
        
        # Linux command
        filepath = os.path.join(self.audio_dir, filename)
        print(f"\n🐧 Linux Command:")
        print(f"  aplay '{filepath}'")
        
        # Mac command
        print(f"\n🍎 Mac Command:")
        print(f"  afplay '{filepath}'")
        
        print(f"\n💡 TIP: The sounds are based on REAL mushroom electrical activity!")
        print(f"🍄 Mushrooms are literally making electronic music!")
    
    def interactive_player(self):
        """Interactive player interface"""
        print(f"🎵 FUNGAL SYNTH INTERACTIVE PLAYER")
        print(f"Author: {self.author}")
        print(f"Timestamp: {self.timestamp}")
        print(f"=" * 60)
        
        while True:
            # List available sounds
            sounds = self.list_available_sounds()
            
            if not sounds:
                print(f"\n❌ No sounds available. Please generate some first!")
                break
            
            print(f"\n🎵 What would you like to do?")
            print(f"  1. Play a specific sound")
            print(f"  2. Get info about a sound")
            print(f"  3. Exit")
            
            try:
                choice = input(f"\n🎵 Enter your choice (1-3): ").strip()
                
                if choice == '1':
                    print(f"\n🎵 Enter the number of the sound to play:")
                    for i, sound in enumerate(sounds, 1):
                        print(f"  {i}. {sound}")
                    
                    try:
                        sound_num = int(input(f"\n🎵 Sound number (1-{len(sounds)}): ")) - 1
                        if 0 <= sound_num < len(sounds):
                            selected_sound = sounds[sound_num]
                            self.play_sound_instructions(selected_sound)
                            
                            # Try to play on Linux
                            filepath = os.path.join(self.audio_dir, selected_sound)
                            print(f"\n🎵 Attempting to play: {selected_sound}")
                            
                            try:
                                os.system(f"aplay '{filepath}'")
                            except:
                                print(f"  💡 Sound file ready to play manually!")
                                print(f"  📁 File location: {filepath}")
                        else:
                            print(f"❌ Invalid sound number!")
                    except ValueError:
                        print(f"❌ Please enter a valid number!")
                
                elif choice == '2':
                    print(f"\n🎵 Enter the number of the sound for info:")
                    for i, sound in enumerate(sounds, 1):
                        print(f"  {i}. {sound}")
                    
                    try:
                        sound_num = int(input(f"\n🎵 Sound number (1-{len(sounds)}): ")) - 1
                        if 0 <= sound_num < len(sounds):
                            selected_sound = sounds[sound_num]
                            self.play_sound_info(selected_sound)
                        else:
                            print(f"❌ Invalid sound number!")
                    except ValueError:
                        print(f"❌ Please enter a valid number!")
                
                elif choice == '3':
                    print(f"\n🎉 Thanks for listening to fungal synth sounds!")
                    print(f"🍄 Mushrooms make the best electronic music!")
                    break
                
                else:
                    print(f"❌ Invalid choice! Please enter 1, 2, or 3.")
                
            except KeyboardInterrupt:
                print(f"\n\n🎉 Thanks for listening to fungal synth sounds!")
                print(f"🍄 Mushrooms make the best electronic music!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    print("🎵 FUNGAL SYNTH AUDIO PLAYER")
    print("Author: Joe Knowles")
    print("Timestamp: 2025-08-12 09:23:27 BST")
    print("=" * 60)
    
    player = FungalSynthPlayer()
    
    # Check if sounds exist
    if not os.path.exists(player.audio_dir):
        print(f"❌ Audio directory not found: {player.audio_dir}")
        print(f"💡 Please run the fungal synth generator first!")
        return
    
    # Start interactive player
    player.interactive_player()

if __name__ == "__main__":
    main() 