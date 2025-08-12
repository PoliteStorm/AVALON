#!/usr/bin/env python3
"""
Fungal Audio Player
Author: Joe Knowles
Timestamp: 2025-08-12 09:23:27 BST
Description: Simple audio player for fungal electrical signals
"""

import numpy as np
import os
from pathlib import Path
import json
import time

class FungalAudioPlayer:
    """Simple audio player for fungal electrical signals."""
    
    def __init__(self):
        self.timestamp = "20250812_092327"
        self.author = "Joe Knowles"
        self.audio_dir = "RESULTS/audio"
        self.sample_rate = 44100
        
    def list_available_sounds(self):
        """List all available audio files."""
        print("🎵 Available Fungal Sounds:")
        print("=" * 50)
        
        if not os.path.exists(self.audio_dir):
            print("❌ No audio directory found. Run fungal_sound_generator.py first.")
            return []
        
        audio_files = []
        for file_path in Path(self.audio_dir).glob("*"):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                audio_files.append((file_path.name, size_mb))
        
        if not audio_files:
            print("❌ No audio files found. Run fungal_sound_generator.py first.")
            return []
        
        # Group by type
        basic_sounds = [f for f, _ in audio_files if "basic_sound" in f]
        fm_sounds = [f for f, _ in audio_files if "frequency_modulated" in f]
        rhythm_sounds = [f for f, _ in audio_files if "rhythm_based" in f]
        relationship_sounds = [f for f, _ in audio_files if "relationship" in f]
        
        print(f"🔊 Basic Sounds ({len(basic_sounds)}):")
        for sound in basic_sounds:
            print(f"  • {sound}")
        
        print(f"\n🎵 Frequency Modulated ({len(fm_sounds)}):")
        for sound in fm_sounds:
            print(f"  • {sound}")
        
        print(f"\n🥁 Rhythm Based ({len(rhythm_sounds)}):")
        for sound in rhythm_sounds:
            print(f"  • {sound}")
        
        print(f"\n🔗 Relationship Sounds ({len(relationship_sounds)}):")
        for sound in relationship_sounds:
            print(f"  • {sound}")
        
        return audio_files
    
    def play_audio_file(self, filename):
        """Play an audio file using available methods."""
        file_path = Path(self.audio_dir) / filename
        
        if not file_path.exists():
            print(f"❌ File not found: {filename}")
            return False
        
        print(f"🎵 Playing: {filename}")
        print(f"📁 File: {file_path}")
        
        # Try to play using different methods
        success = False
        
        # Method 1: Try using pygame (if available)
        success = self.play_with_pygame(file_path)
        if success:
            return True
        
        # Method 2: Try using simpleaudio (if available)
        success = self.play_with_simpleaudio(file_path)
        if success:
            return True
        
        # Method 3: Try using pyaudio (if available)
        success = self.play_with_pyaudio(file_path)
        if success:
            return True
        
        # Method 4: Convert and provide instructions
        self.provide_playback_instructions(file_path)
        return False
    
    def play_with_pygame(self, file_path):
        """Try to play audio using pygame."""
        try:
            import pygame
            pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=1)
            
            # Load and play the audio
            pygame.mixer.music.load(str(file_path))
            pygame.mixer.music.play()
            
            print("  🔊 Playing with pygame...")
            print("  ⏸️  Press Enter to stop...")
            
            input()
            pygame.mixer.music.stop()
            pygame.mixer.quit()
            
            return True
            
        except ImportError:
            print("  ℹ️  pygame not available")
            return False
        except Exception as e:
            print(f"  ❌ pygame error: {e}")
            return False
    
    def play_with_simpleaudio(self, file_path):
        """Try to play audio using simpleaudio."""
        try:
            import simpleaudio as sa
            
            # Load audio data
            if file_path.suffix == '.npy':
                audio_data = np.load(file_path)
            else:
                # Try to load as raw PCM
                with open(file_path, 'rb') as f:
                    audio_data = np.frombuffer(f.read(), dtype=np.int16)
            
            # Convert to float and normalize
            audio_data = audio_data.astype(np.float32) / 32767.0
            
            # Play audio
            play_obj = sa.play_buffer(audio_data, 1, 2, self.sample_rate)
            print("  🔊 Playing with simpleaudio...")
            print("  ⏸️  Audio will play to completion...")
            
            play_obj.wait_done()
            return True
            
        except ImportError:
            print("  ℹ️  simpleaudio not available")
            return False
        except Exception as e:
            print(f"  ❌ simpleaudio error: {e}")
            return False
    
    def play_with_pyaudio(self, file_path):
        """Try to play audio using pyaudio."""
        try:
            import pyaudio
            
            # Load audio data
            if file_path.suffix == '.npy':
                audio_data = np.load(file_path)
            else:
                # Try to load as raw PCM
                with open(file_path, 'rb') as f:
                    audio_data = np.frombuffer(f.read(), dtype=np.int16)
            
            # Convert to float and normalize
            audio_data = audio_data.astype(np.float32) / 32767.0
            
            # Initialize PyAudio
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32,
                           channels=1,
                           rate=self.sample_rate,
                           output=True)
            
            # Play audio
            print("  🔊 Playing with pyaudio...")
            print("  ⏸️  Audio will play to completion...")
            
            stream.write(audio_data.tobytes())
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            return True
            
        except ImportError:
            print("  ℹ️  pyaudio not available")
            return False
        except Exception as e:
            print(f"  ❌ pyaudio error: {e}")
            return False
    
    def provide_playback_instructions(self, file_path):
        """Provide instructions for manual playback."""
        print(f"\n📋 Manual Playback Instructions:")
        print(f"=" * 50)
        
        if file_path.suffix == '.npy':
            print(f"1. Load the .npy file in Python:")
            print(f"   import numpy as np")
            print(f"   data = np.load('{file_path.name}')")
            print(f"   print(f'Data shape: {{data.shape}}')")
            print(f"   print(f'Data range: {{data.min():.3f}} to {{data.max():.3f}}')")
        
        print(f"\n2. Convert to WAV format:")
        print(f"   import soundfile as sf")
        print(f"   sf.write('output.wav', data, {self.sample_rate})")
        
        print(f"\n3. Play with any audio software:")
        print(f"   • VLC Media Player")
        print(f"   • Audacity")
        print(f"   • Windows Media Player")
        print(f"   • macOS QuickTime")
        
        print(f"\n4. Or use Python audio libraries:")
        print(f"   pip install pygame simpleaudio pyaudio")
    
    def interactive_player(self):
        """Interactive audio player interface."""
        print("🎵 Fungal Audio Interactive Player")
        print("Author: Joe Knowles")
        print("Timestamp: 2025-08-12 09:23:27 BST")
        print("=" * 60)
        
        while True:
            print(f"\n🎵 FUNGAL AUDIO PLAYER MENU")
            print("=" * 40)
            print("1. List available sounds")
            print("2. Play a specific sound")
            print("3. Play all sounds sequentially")
            print("4. Show playback instructions")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                self.list_available_sounds()
                
            elif choice == "2":
                audio_files = self.list_available_sounds()
                if audio_files:
                    print(f"\nEnter the filename to play:")
                    filename = input("Filename: ").strip()
                    if filename:
                        self.play_audio_file(filename)
                
            elif choice == "3":
                audio_files = self.list_available_sounds()
                if audio_files:
                    print(f"\n🎵 Playing all sounds sequentially...")
                    for filename, _ in audio_files:
                        print(f"\n🔊 Playing: {filename}")
                        self.play_audio_file(filename)
                        time.sleep(1)  # Brief pause between sounds
                
            elif choice == "4":
                print(f"\n📋 PLAYBACK INSTRUCTIONS")
                print("=" * 40)
                print("The fungal sounds are saved in multiple formats:")
                print("• .npy files: Numpy arrays (easiest to work with)")
                print("• .raw files: Raw PCM data (for audio software)")
                print("• .json files: Metadata and file information")
                print("\nTo play the sounds, you can:")
                print("1. Use this interactive player")
                print("2. Convert to WAV format and use any media player")
                print("3. Use Python audio libraries directly")
                
            elif choice == "5":
                print("👋 Goodbye! Thanks for listening to the mushrooms!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-5.")

def main():
    """Main function for the audio player."""
    player = FungalAudioPlayer()
    
    # Check if audio files exist
    if not os.path.exists(player.audio_dir):
        print("❌ No audio files found!")
        print("📋 Please run fungal_sound_generator.py first to create sounds.")
        return
    
    # Start interactive player
    player.interactive_player()

if __name__ == "__main__":
    main() 