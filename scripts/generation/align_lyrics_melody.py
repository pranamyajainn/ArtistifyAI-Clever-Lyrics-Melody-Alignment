import os
import copy
from music21 import converter, stream, note, chord, metadata, instrument, midi, meter, volume, pitch

# ✅ File Paths
INSTRUMENTAL_MIDI = "output/full_multi_instrument_song.mid"  # Full-length instrumental
LYRICS_FILE = "output/generated_lyrics.txt"  # AI-generated lyrics
FINAL_MIDI = "output/final_song_with_lyrics.mid"  # Final merged song
XML_OUTPUT_FILE = "output/final_song_with_lyrics.musicxml"  # For MuseScore visualization

# ✅ Load MIDI file
def load_midi_file(filename):
    if not os.path.exists(filename):
        print(f"❌ ERROR: MIDI file '{filename}' not found!")
        exit()
    print(f"🎵 Loading MIDI file: {filename}")
    return converter.parse(filename)

# ✅ Load Lyrics
def load_generated_lyrics(filename):
    if not os.path.exists(filename):
        print(f"❌ ERROR: Lyrics file '{filename}' not found!")
        exit()
    with open(filename, "r", encoding="utf-8") as f:
        lyrics = f.read().strip().split()
    if not lyrics:
        print("❌ ERROR: Lyrics file is empty!")
        exit()
    print(f"📜 Loaded {len(lyrics)} words from lyrics file")
    return lyrics

# ✅ Create a completely new vocal track
def create_new_vocal_track(instrumental_score, lyrics):
    print("🔄 Creating new vocal track from scratch...")
    
    # Create new vocal part
    vocal_part = stream.Part()
    vocal_part.partName = "Vocals"
    
    # Try to find an existing melody to follow
    highest_notes = []
    for part in instrumental_score.parts:
        notes_list = [n for n in part.flatten().notes if isinstance(n, note.Note)]
        if notes_list:
            highest_notes.extend(notes_list)
    
    # Sort by pitch (highest to lowest)
    highest_notes.sort(key=lambda n: n.pitch.midi, reverse=True)
    
    # If we found notes, use them as a guide
    if highest_notes:
        print(f"✅ Found {len(highest_notes)} notes to base vocal melody on")
        
        # Take a subset of these notes (to match our lyrics)
        melody_notes = highest_notes[:min(len(lyrics)*2, len(highest_notes))]
        
        # Add time signature
        ts = meter.TimeSignature('4/4')
        vocal_part.insert(0, ts)
        
        # Add vocal instrument (try a standard choir sound - GM program 53)
        vox = instrument.Instrument()
        vox.midiProgram = 53  # Choir Aahs in General MIDI
        vocal_part.insert(0, vox)
        
        # Create notes with lyrics
        for i, lyric in enumerate(lyrics):
            if i < len(melody_notes):
                # Use the melody note's pitch and timing
                n = copy.deepcopy(melody_notes[i])
                n.lyric = lyric
                
                # Ensure strong volume
                n.volume.velocity = 100
                
                vocal_part.append(n)
    else:
        print("⚠️ No melody notes found, creating basic vocal line...")
        
        # Create a simple melodic line with C4-G4 range
        ts = meter.TimeSignature('4/4')
        vocal_part.insert(0, ts)
        
        vox = instrument.Instrument()
        vox.midiProgram = 53  # Choir Aahs
        vocal_part.insert(0, vox)
        
        # Create a basic scale for notes
        note_pitches = ['C4', 'D4', 'E4', 'F4', 'G4', 'F4', 'E4', 'D4']
        
        # Add notes with lyrics
        for i, lyric in enumerate(lyrics):
            p = note_pitches[i % len(note_pitches)]
            n = note.Note(p)
            n.quarterLength = 1.0
            n.lyric = lyric
            n.volume.velocity = 100
            vocal_part.append(n)
    
    # Try to organize into measures
    try:
        return vocal_part.makeMeasures()
    except Exception as e:
        print(f"⚠️ Warning: {e}")
        return vocal_part

# ✅ Save direct karaoke MIDI file
def save_karaoke_midi(instrumental_score, lyrics, output_file):
    print("🎤 Creating direct karaoke MIDI file...")
    
    # Most MIDI karaoke files use channel 4 (index 3) for lyrics by convention
    KARAOKE_CHANNEL = 3
    
    # Create a new stream
    s = copy.deepcopy(instrumental_score)
    
    # Find a good melody part to attach lyrics
    melody_part = None
    highest_avg = 0
    
    for part in s.parts:
        notes = [n for n in part.flatten().notes if isinstance(n, note.Note)]
        if not notes:
            continue
        
        avg_pitch = sum(n.pitch.midi for n in notes) / len(notes)
        if avg_pitch > highest_avg:
            highest_avg = avg_pitch
            melody_part = part
    
    if not melody_part:
        print("⚠️ No suitable melody part found, using first part")
        if s.parts:
            melody_part = s.parts[0]
        else:
            print("❌ ERROR: No parts found in the score!")
            return False
    
    # Get notes from the melody part
    melody_notes = [n for n in melody_part.flatten().notes if isinstance(n, note.Note)]
    
    if len(melody_notes) < len(lyrics):
        print(f"⚠️ Not enough notes ({len(melody_notes)}) for lyrics ({len(lyrics)})")
        lyrics = lyrics[:len(melody_notes)]
    elif len(melody_notes) > len(lyrics):
        print(f"⚠️ More notes ({len(melody_notes)}) than lyrics ({len(lyrics)})")
        # Repeat lyrics to fill
        lyrics = (lyrics * (len(melody_notes) // len(lyrics) + 1))[:len(melody_notes)]
    
    # Attach lyrics to the melody notes
    for i, n in enumerate(melody_notes):
        if i < len(lyrics):
            n.lyric = lyrics[i]
    
    # Ensure melody part is on the karaoke channel
    for elem in melody_part.flatten():
        if isinstance(elem, instrument.Instrument):
            elem.midiChannel = KARAOKE_CHANNEL
    
    try:
        # Save the modified MIDI file
        s.write('midi', fp=output_file)
        print(f"✅ Karaoke MIDI file saved: {output_file}")
        return True
    except Exception as e:
        print(f"❌ Error saving karaoke MIDI: {e}")
        return False

# ✅ Create a separate lyrics-only MIDI file
def create_lyrics_only_midi(lyrics, output_file):
    print("📝 Creating lyrics-only MIDI file...")
    
    s = stream.Score()
    p = stream.Part()
    
    # Add basic time signature and instrument
    p.insert(0, meter.TimeSignature('4/4'))
    
    # Create a choir instrument
    choir = instrument.Instrument()
    choir.midiProgram = 53  # Choir Aahs
    p.insert(0, choir)
    
    # Create notes with lyrics on a simple melody
    base_pitches = ['C4', 'D4', 'E4', 'F4', 'G4']
    
    for i, word in enumerate(lyrics):
        # Create a note
        n = note.Note(base_pitches[i % len(base_pitches)])
        n.quarterLength = 1.0
        n.lyric = word
        n.volume.velocity = 100
        p.append(n)
    
    # Add the part to the score
    s.insert(0, p)
    
    # Save as MIDI
    try:
        s.write('midi', fp=output_file)
        print(f"✅ Lyrics-only MIDI saved: {output_file}")
        return True
    except Exception as e:
        print(f"❌ Error saving lyrics-only MIDI: {e}")
        return False

# ✅ Main Execution
if __name__ == "__main__":
    print("🚀 Running Simplified Vocal Script...")

    # Step 1: Load MIDI and Lyrics
    instrumental = load_midi_file(INSTRUMENTAL_MIDI)
    lyrics = load_generated_lyrics(LYRICS_FILE)

    # Try multiple approaches to ensure vocals work
    
    # Approach 1: Direct karaoke MIDI
    karaoke_output = "output/karaoke_version.mid"
    save_karaoke_midi(instrumental, lyrics, karaoke_output)
    
    # Approach 2: Create standalone lyrics file
    lyrics_only_output = "output/lyrics_only.mid"
    create_lyrics_only_midi(lyrics, lyrics_only_output)
    
    # Approach 3: Traditional music21 approach
    vocal_track = create_new_vocal_track(instrumental, lyrics)
    
    # Create a combined score
    final_score = stream.Score()
    
    # Add the vocal part first (for prominence)
    final_score.insert(0, vocal_track)
    
    # Add all instrumental parts
    for part in instrumental.parts:
        final_score.insert(0, part)
    
    # Save as MIDI and MusicXML
    try:
        final_score.write("midi", fp=FINAL_MIDI)
        final_score.write("musicxml", fp=XML_OUTPUT_FILE)
        print(f"✅ Final MIDI saved: {FINAL_MIDI}")
        print(f"✅ MusicXML saved: {XML_OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ Error saving final files: {e}")
    
    # Print guidance
    print("\n🎬 Process complete! We've created multiple files to ensure vocals work:")
    print(f"1. {karaoke_output} - Standard karaoke MIDI format")
    print(f"2. {lyrics_only_output} - Lyrics-only MIDI (to verify lyrics are encoded properly)")
    print(f"3. {FINAL_MIDI} - Full song with embedded vocal track")
    print(f"4. {XML_OUTPUT_FILE} - MusicXML for viewing in notation software\n")
    
    print("🔍 Troubleshooting tips:")
    print("- Open the MusicXML file in MuseScore to see if lyrics are visible")
    print("- Try a karaoke-specific MIDI player like VanBasco's Karaoke Player")
    print("- Some MIDI players don't display lyrics - try Windows Media Player or QuickTime")
    print("- The lyrics-only file should help determine if your player supports MIDI lyrics")