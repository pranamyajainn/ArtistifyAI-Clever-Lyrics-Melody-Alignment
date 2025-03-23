import os
# Disable GPU to avoid CUDA errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.keras.models import load_model
import numpy as np
from music21 import stream, note, midi, instrument, tempo as music21_tempo, chord, duration, metadata
import os
import random

# Load the trained melody generator model
try:
    melody_model = load_model("models/melody_generator.h5")
    print("✅ Melody Model Loaded Successfully!")
except:
    print("ℹ️ Model not found, will use scale-based generation only")
    melody_model = None

def generate_melody(model, start_sequence, num_notes=100):
    """
    Generate a melody using the trained model.
    
    Args:
    - model: Trained melody generator model.
    - start_sequence: Initial sequence to start generation.
    - num_notes: Number of notes to generate.
    
    Returns:
    - Generated melody sequence.
    """
    generated_notes = start_sequence[:]
    
    for _ in range(num_notes):
        input_seq = np.array(generated_notes[-50:]).reshape(1, 50, 1)  # Use last 50 notes as input
        predicted_note = np.argmax(model.predict(input_seq, verbose=0))  # Predict next note
        generated_notes.append(predicted_note)
    
    return generated_notes

def create_musical_melody(length=32, scale_type="major", start_note=60, 
                          complexity=1.0, rhythm_pattern=None):
    """
    Create a musical melody based on actual musical scales
    
    Args:
    - length: Number of notes to generate
    - scale_type: Type of scale to use (major, minor, pentatonic, blues)
    - start_note: MIDI note to start with
    - complexity: Controls how complex the melody should be (0.0-2.0)
    - rhythm_pattern: Optional predefined rhythm pattern
    
    Returns:
    - List of (MIDI note numbers, duration) pairs in a proper musical scale
    """
    # Define different scales (MIDI offsets from root)
    scales = {
        "major": [0, 2, 4, 5, 7, 9, 11, 12],  # Major scale
        "minor": [0, 2, 3, 5, 7, 8, 10, 12],  # Natural minor scale
        "pentatonic": [0, 2, 4, 7, 9, 12],    # Major pentatonic
        "blues": [0, 3, 5, 6, 7, 10, 12],     # Blues scale
        "dorian": [0, 2, 3, 5, 7, 9, 10, 12]  # Dorian mode
    }
    
    # Default to major if scale type not found
    current_scale = scales.get(scale_type, scales["major"])
    
    # Calculate the root note (C = 60, etc.)
    root_note = start_note % 12
    octave = start_note // 12 - 1
    
    # Build full scale across multiple octaves
    full_scale = []
    for octave_offset in range(-1, 3):  # -1 octave to +2 octaves
        for note_offset in current_scale:
            full_scale.append(root_note + note_offset + ((octave + octave_offset) * 12))
    
    # Filter to keep only notes in a good range (G3 to C6)
    full_scale = [note for note in full_scale if 55 <= note <= 84]
    
    # Define rhythm patterns (quarter note = 1.0)
    if not rhythm_pattern:
        rhythm_patterns = [
            [1.0, 1.0, 1.0, 1.0],  # Simple quarter notes
            [0.5, 0.5, 1.0, 0.5, 0.5, 1.0],  # Eighth-quarter mix
            [0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 1.0],  # More complex
            [0.5, 0.25, 0.25, 1.0, 0.5, 0.5],  # Syncopated
            [1.5, 0.5, 1.0, 1.0]  # Dotted quarter + eighth
        ]
        # Select a rhythm pattern based on complexity
        pattern_index = min(int(complexity * len(rhythm_patterns)), len(rhythm_patterns) - 1)
        rhythm_pattern = rhythm_patterns[pattern_index]
    
    # Generate melody using the scale
    melody = []
    current_note = start_note
    
    # Interval probabilities based on complexity
    if complexity < 0.7:
        intervals = [-1, 0, 0, 1, 1]  # Simple - small steps
    elif complexity < 1.3:
        intervals = [-2, -1, -1, 0, 0, 1, 1, 2]  # Medium - larger steps
    else:
        intervals = [-3, -2, -1, 0, 1, 2, 3, 4]  # Complex - larger jumps
    
    # Start with the provided note
    melody.append((current_note, rhythm_pattern[0]))
    
    rhythm_idx = 1
    for i in range(1, length):
        # Get the last note (without duration)
        last_note = melody[-1][0]
        
        # Decide the interval to the next note
        interval = random.choice(intervals)
        
        # Find the current position in the scale
        try:
            current_pos = full_scale.index(last_note)
        except ValueError:
            # If last_note is None, choose a default position (e.g., 0)
            if last_note is None:
                current_pos = 0
            else:
                current_pos = min(range(len(full_scale)), key=lambda i: abs(full_scale[i] - last_note))

        
        # Calculate new position with boundary checks
        new_pos = max(0, min(len(full_scale) - 1, current_pos + interval))
        
        # Get the new note
        current_note = full_scale[new_pos]
        
        # Get duration from rhythm pattern (cycling)
        duration = rhythm_pattern[rhythm_idx % len(rhythm_pattern)]
        rhythm_idx += 1
        
        # Occasionally add a rest with probability based on complexity
        if random.random() < 0.05 * complexity:
            melody.append((None, duration))  # Rest
        else:
            melody.append((current_note, duration))
    
    return melody

def create_chord_progression(key=60, scale_type="major", length=4, pattern=None):
    """
    Create a chord progression based on a key and scale
    
    Args:
    - key: Root note of the key (60 = C)
    - scale_type: Type of scale
    - length: Number of chords
    - pattern: Optional predefined chord pattern (list of scale degrees)
    
    Returns:
    - List of chord notes and durations
    """
    # Common chord progressions by scale degrees (1-based)
    progressions = {
        "basic": [1, 4, 5, 1],         # I-IV-V-I
        "pop": [1, 5, 6, 4],           # I-V-vi-IV
        "blues": [1, 4, 1, 5, 4, 1],   # I-IV-I-V-IV-I
        "jazz": [2, 5, 1, 6],          # ii-V-I-vi
        "epic": [1, 5, 6, 3, 4, 1, 4, 5]  # I-V-vi-iii-IV-I-IV-V
    }
    
    # Scale degrees to build chords (0-based)
    scale_degrees = {
        "major": [0, 2, 4, 5, 7, 9, 11],   # Major scale
        "minor": [0, 2, 3, 5, 7, 8, 10]    # Natural minor
    }
    
    # Chord types by scale degree (major or minor triads)
    chord_types = {
        "major": ["M", "m", "m", "M", "M", "m", "dim"],  # Major scale chord types
        "minor": ["m", "dim", "M", "m", "m", "M", "M"]   # Minor scale chord types
    }
    
    # Get the scale
    scale = scale_degrees.get(scale_type, scale_degrees["major"])
    
    # Select a progression or use the provided pattern
    if not pattern:
        progression_type = random.choice(list(progressions.keys()))
        pattern = progressions[progression_type]
    
    # Adjust pattern to be 0-based
    pattern = [degree - 1 for degree in pattern]
    
    # Create the chord progression
    progression = []
    root_note = key % 12
    octave = key // 12 - 1
    
    # Duration for each chord (in quarter notes)
    chord_duration = 4.0  # Default to one measure per chord (4 beats)
    
    for degree in pattern[:length]:  # Limit to requested length
        # Get the root note of this chord
        chord_root = root_note + scale[degree % len(scale)] + (octave * 12)
        
        # Determine chord type (major, minor, diminished)
        chord_type = chord_types[scale_type][degree % len(scale)]
        
        # Build the chord
        chord_notes = [chord_root]  # Root note
        
        # Add third
        if chord_type == "M":
            chord_notes.append(chord_root + 4)  # Major third
        elif chord_type in ["m", "dim"]:
            chord_notes.append(chord_root + 3)  # Minor third
            
        # Add fifth
        if chord_type in ["M", "m"]:
            chord_notes.append(chord_root + 7)  # Perfect fifth
        elif chord_type == "dim":
            chord_notes.append(chord_root + 6)  # Diminished fifth
            
        # Add seventh for more complex chords (optional)
        if random.random() > 0.6:
            if chord_type == "M":
                chord_notes.append(chord_root + 11)  # Major seventh
            elif chord_type == "m":
                chord_notes.append(chord_root + 10)  # Minor seventh
            elif chord_type == "dim":
                chord_notes.append(chord_root + 9)   # Diminished seventh
                
        progression.append((chord_notes, chord_duration))
    
    return progression

def create_bassline(chord_progression, complexity=1.0):
    """
    Create a bassline that follows the chord progression
    
    Args:
    - chord_progression: List of chord notes and durations
    - complexity: Controls pattern complexity
    
    Returns:
    - List of (note, duration) pairs
    """
    bassline = []
    
    for chord, total_duration in chord_progression:
        root_note = chord[0]
        
        # Different bassline patterns based on complexity
        if complexity < 0.7:
            # Simple - just the root note
            bassline.append((root_note - 12, total_duration))
            
        elif complexity < 1.3:
            # Medium - root and fifth pattern
            bassline.append((root_note - 12, total_duration / 2))
            bassline.append((root_note - 5, total_duration / 2))
            
        else:
            # Complex - walking bass pattern
            fifth = root_note - 5
            bassline.append((root_note - 12, total_duration / 4))
            bassline.append((root_note - 5, total_duration / 4))
            bassline.append((root_note - 7, total_duration / 4))
            bassline.append((root_note - 10, total_duration / 4))
    
    return bassline

def create_drum_pattern(length=4, style="basic", intensity=1.0):
    """
    Create a drum pattern
    
    Args:
    - length: Number of measures
    - style: Drum style (basic, rock, jazz, etc.)
    - intensity: Controls fill frequency and complexity
    
    Returns:
    - Dictionary of drum notes by type
    """
    # MIDI notes for different drum sounds
    drums = {
        "kick": 36,     # Bass drum
        "snare": 38,    # Snare drum
        "hihat": 42,    # Closed hi-hat
        "openhat": 46,  # Open hi-hat
        "tom1": 47,     # Low-mid tom
        "tom2": 45,     # Low tom
        "crash": 49,    # Crash cymbal
        "ride": 51      # Ride cymbal
    }
    
    # Basic patterns (16 steps per measure = 16th notes)
    patterns = {
        "basic": {
            "kick":   [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            "snare":  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            "hihat":  [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        },
        "rock": {
            "kick":   [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
            "snare":  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            "hihat":  [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        },
        "funk": {
            "kick":   [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            "snare":  [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            "hihat":  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        },
        "jazz": {
            "kick":   [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            "ride":   [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "hihat":  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        }
    }
    
    # Create fills based on intensity
    def create_fill(intensity):
        fill = {
            "kick": [0] * 16,
            "snare": [0] * 16,
            "tom1": [0] * 16,
            "tom2": [0] * 16,
            "crash": [0] * 16
        }
        
        # More intense fills have more notes
        note_count = int(5 + intensity * 6)
        
        for _ in range(note_count):
            drum_type = random.choice(["kick", "snare", "tom1", "tom2"])
            position = random.randint(0, 15)
            fill[drum_type][position] = 1
        
        # Add crash at the end
        fill["crash"][-1] = 1
        
        return fill
    
    # Choose a pattern
    if style not in patterns:
        style = "basic"
    pattern = patterns[style]
    
    # Expand pattern for desired length
    drum_notes = {drum_type: [] for drum_type in drums.keys()}
    
    for measure in range(length):
        # Determine if this measure should have a fill
        has_fill = (measure == length - 1) or (random.random() < 0.2 * intensity and measure > 0)
        
        if has_fill and measure == length - 1:
            # Final measure fill
            fill = create_fill(intensity)
            for drum_type, positions in fill.items():
                if drum_type in drum_notes:
                    for pos, hit in enumerate(positions):
                        if hit:
                            # Duration: 16th note (0.25 quarter notes)
                            drum_notes[drum_type].append((drums[drum_type], 0.25))
                        else:
                            # Rest
                            drum_notes[drum_type].append((None, 0.25))
        else:
            # Regular pattern
            for drum_type in pattern.keys():
                positions = pattern[drum_type]
                for pos, hit in enumerate(positions):
                    if hit:
                        # Duration: 16th note (0.25 quarter notes)
                        drum_notes[drum_type].append((drums[drum_type], 0.25))
                    else:
                        # Rest
                        drum_notes[drum_type].append((None, 0.25))
    
    return drum_notes

def generate_multi_instrument_song(filename="output/full_song.mid", bpm=100):
    """
    Generate a complete song with multiple instruments
    
    Args:
    - filename: Output MIDI file name
    - bpm: Song tempo in BPM
    
    Returns:
    - Path to the generated MIDI file
    """
    # Ensure output folder exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Create a music21 stream
    song = stream.Score()
    
    # Set metadata
    song.insert(0, metadata.Metadata())
    song.metadata.title = "Generated Multi-Instrument Song"
    song.metadata.composer = "AI Composer"
    
    # Set the tempo - FIXED: Use music21_tempo instead
    song.insert(0, music21_tempo.MetronomeMark(number=bpm))
    
    # Song structure parameters
    song_key = 60  # C
    song_scale = "major"
    song_structure = ["intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"]
    section_lengths = {
        "intro": 4,
        "verse": 8,
        "chorus": 8,
        "bridge": 4,
        "outro": 4
    }
    
    # Create instrumental parts
    parts = {
        "lead": {
            "instrument": instrument.Piano(),
            "part": stream.Part()
        },
        "pad": {
            "instrument": instrument.ElectricGuitar(),

            "part": stream.Part()
        },
        "bass": {
            "instrument": instrument.ElectricBass(),
            "part": stream.Part()
        },
        "rhythm": {
            "instrument": instrument.ElectricGuitar(),
            "part": stream.Part()
        },
        "drums": {
            "instrument": instrument.Dulcimer(),
            "part": stream.Part()
        }
    }
    
    # Add instruments to parts
    for part_info in parts.values():
        part_info["part"].append(part_info["instrument"])
    
    # Process each section of the song
    current_measure = 0
    
    for section in song_structure:
        section_length = section_lengths[section]
        # Adjust complexity based on section
        if section in ["intro", "outro"]:
            complexity = 0.7
        elif section == "verse":
            complexity = 1.0
        elif section == "chorus":
            complexity = 1.3
        elif section == "bridge":
            complexity = 1.5
        
        # Create chord progression for this section
        if section == "intro":
            progression = create_chord_progression(key=song_key, scale_type=song_scale, 
                                                  length=section_length, pattern=[1, 5, 6, 4])
        elif section == "verse":
            progression = create_chord_progression(key=song_key, scale_type=song_scale, 
                                                  length=section_length, pattern=[1, 5, 6, 4, 1, 5, 4, 5])
        elif section == "chorus":
            progression = create_chord_progression(key=song_key, scale_type=song_scale, 
                                                  length=section_length, pattern=[1, 4, 6, 5, 1, 4, 6, 5])
        elif section == "bridge":
            progression = create_chord_progression(key=song_key, scale_type=song_scale, 
                                                  length=section_length, pattern=[6, 5, 4, 5])
        elif section == "outro":
            progression = create_chord_progression(key=song_key, scale_type=song_scale, 
                                                  length=section_length, pattern=[1, 5, 6, 1])
        
        # Create bassline from chord progression
        bassline = create_bassline(progression, complexity=complexity)
        
        # Create drum pattern
        if section in ["intro", "outro"]:
            drum_style = "basic"
        elif section == "verse":
            drum_style = "rock"
        elif section == "chorus":
            drum_style = "funk"
        elif section == "bridge":
            drum_style = "jazz"
            
        drum_notes = create_drum_pattern(length=section_length, style=drum_style, intensity=complexity)
        
        # Create lead melody - use either the model or musical approach
        if melody_model is not None and section in ["verse", "chorus"]:
            # Use the trained model for melody generation
            # Generate a starting sequence in our desired range (C4 to C5)
            random_start = np.random.randint(60, 72, size=50).tolist()
            
            # Generate a melody
            generated_melody = generate_melody(melody_model, random_start, num_notes=section_length * 16)
            print(f"✅ Model-based melody generated for {section}")
            
            # Map to a proper musical range (G3 to C6)
            min_note, max_note = 55, 84  # G3 to C6
            
            # Find the min and max of the original sequence
            orig_min = min(generated_melody)
            orig_max = max(generated_melody)
            orig_range = max(1, orig_max - orig_min)  # Avoid division by zero
            
            # Map each note to the new range and convert to melody format
            melody = []
            for i, note_val in enumerate(generated_melody):
                # Normalize to 0-1 range
                normalized = (note_val - orig_min) / orig_range
                # Map to target range
                mapped_note = min_note + round(normalized * (max_note - min_note))
                
                # Every 4 notes, decide duration based on complexity
                if i % 4 == 0:
                    if complexity < 0.7:
                        duration = 1.0  # Quarter note
                    elif complexity < 1.3:
                        duration = 0.5  # Eighth note
                    else:
                        duration = random.choice([0.25, 0.5, 0.75, 1.0])  # Mixed durations
                
                melody.append((mapped_note, duration))
                
        else:
            # Scale-based melody generation
            if section == "intro" or section == "outro":
                # Simple melody for intro/outro
                melody_length = section_length * 16  # 16 sixteenth notes per measure
                melody = create_musical_melody(length=melody_length, scale_type=song_scale, 
                                             start_note=song_key + 12, complexity=complexity)
            else:
                # More notes for other sections
                melody_length = section_length * 16
                melody = create_musical_melody(length=melody_length, scale_type=song_scale, 
                                             start_note=song_key + 12, complexity=complexity)
        
        # Create rhythm guitar/keyboard part based on chord progression
        rhythm_part = []
        for chord_notes, chord_duration in progression:
            # Break chord into shorter rhythmic patterns based on section
            if section == "verse":
                # Quarter note strums
                for _ in range(int(chord_duration)):
                    rhythm_part.append((chord_notes, 1.0))
            elif section == "chorus":
                # Eighth note strums
                for _ in range(int(chord_duration * 2)):
                    rhythm_part.append((chord_notes, 0.5))
            else:
                # Whole note for intro, bridge, outro
                rhythm_part.append((chord_notes, chord_duration))
        
        # Create pad/synthesizer part (long sustained chords)
        pad_part = []
        for chord_notes, chord_duration in progression:
            # Always hold full duration
            pad_part.append((chord_notes, chord_duration))
        
        # Add instrument silence during breaks
        # Let's create dynamic arrangement by temporarily silencing some instruments
        
        # INTRO: Start with just piano
        if section == "intro":
            # Silence all but piano for first half of intro
            if section_length > 2:
                for i in range(2 * 16):  # 2 measures * 16 16th notes
                    if i < len(melody):
                        melody[i] = (None, melody[i][1])  # Replace with rest
        
        # VERSE: Sometimes drop drums for a measure
        elif section == "verse":
            if random.random() > 0.7:
                # Choose a measure to break
                break_measure = random.randint(0, section_length - 1)
                for drum_type in drum_notes:
                    for i in range(break_measure * 16, (break_measure + 1) * 16):
                        if i < len(drum_notes[drum_type]):
                            drum_notes[drum_type][i] = (None, drum_notes[drum_type][i][1])
        
        # BRIDGE: Drop everything except piano and bass for first measures
        elif section == "bridge":
            # Silence rhythm and drums for first measure
            for i in range(min(16, len(rhythm_part))):
                rhythm_part[i] = (None, rhythm_part[i][1])  # Replace with rest
            
            for drum_type in drum_notes:
                for i in range(16):  # First measure
                    if i < len(drum_notes[drum_type]):
                        drum_notes[drum_type][i] = (None, drum_notes[drum_type][i][1])
        
        # OUTRO: Gradually remove instruments
        elif section == "outro":
            # Keep only piano and bass in last measure
            last_measure_start = (section_length - 1) * 16
            
            for i in range(last_measure_start, last_measure_start + 16):
                if i < len(melody):
                    # Keep melody but make it softer
                    pass
                    
            for drum_type in drum_notes:
                for i in range(last_measure_start, last_measure_start + 16):
                    if i < len(drum_notes[drum_type]):
                        if drum_type != "crash":  # Keep final crash
                            drum_notes[drum_type][i] = (None, drum_notes[drum_type][i][1])
            
            for i in range(section_length - 1, len(rhythm_part)):
                rhythm_part[i] = (None, rhythm_part[i][1])
        
        # Add melody notes to lead part
        current_time = current_measure * 4  # 4 beats per measure
        for note_value, note_duration in melody:
            if note_value is not None:
                new_note = note.Note(note_value)
                new_note.duration.quarterLength = note_duration
                new_note.volume.velocity = random.randint(90, 110)
                parts["lead"]["part"].insert(current_time, new_note)
            current_time += note_duration
        
        # Add rhythm notes
        current_time = current_measure * 4
        for chord_notes, chord_duration in rhythm_part:
            if chord_notes is not None:
                # Create a chord object
                new_chord = chord.Chord(chord_notes)
                new_chord.duration.quarterLength = chord_duration
                new_chord.volume.velocity = random.randint(70, 85)
                parts["rhythm"]["part"].insert(current_time, new_chord)
            current_time += chord_duration
        
        # Add pad/synth notes
        current_time = current_measure * 4
        for chord_notes, chord_duration in pad_part:
            if chord_notes is not None:
                # Create a chord object
                new_chord = chord.Chord(chord_notes)
                new_chord.duration.quarterLength = chord_duration
                new_chord.volume.velocity = random.randint(60, 75)
                parts["pad"]["part"].insert(current_time, new_chord)
            current_time += chord_duration
        
        # Add bass notes
        current_time = current_measure * 4
        for note_value, note_duration in bassline:
            if note_value is not None:
                new_note = note.Note(note_value)
                new_note.duration.quarterLength = note_duration
                new_note.volume.velocity = random.randint(90, 110)
                parts["bass"]["part"].insert(current_time, new_note)
            current_time += note_duration
        
        # Add drum notes
        for drum_type, drum_sequence in drum_notes.items():
            current_time = current_measure * 4
            for note_value, note_duration in drum_sequence:
                if note_value is not None:
                    new_note = note.Note(note_value)
                    new_note.duration.quarterLength = note_duration
                    new_note.volume.velocity = random.randint(80, 100)
                    parts["drums"]["part"].insert(current_time, new_note)
                current_time += note_duration
        
        # Update current measure
        current_measure += section_length
    
    # Add all parts to the score
    for part_info in parts.values():
        song.insert(0, part_info["part"])
    
    # Save as MIDI file
    song.write("midi", fp=filename)
    print(f"✅ Multi-instrument song saved as {filename}")
    
    return filename

if __name__ == "__main__":
    # Generate the full song
    output_file = generate_multi_instrument_song(filename="output/full_multi_instrument_song.mid", bpm=110)
    print(f"Song generation complete! File saved to: {output_file}")