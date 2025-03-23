import numpy as np
import pickle
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # ✅ Force TensorFlow to use CPU only

# ✅ Fix: Ensure correct path for processed notes file
notes_file = os.path.join("data", "processed_notes.pkl")

# Load processed notes
with open(notes_file, "rb") as f:
    notes = pickle.load(f)

# Create a mapping of notes to integers
unique_notes = sorted(set(notes))
note_to_int = {note: number for number, note in enumerate(unique_notes)}
int_to_note = {number: note for note, number in note_to_int.items()}

# Convert notes to integer sequences
sequence_length = 50  # Number of notes per input sequence
input_sequences = []
output_notes = []

for i in range(len(notes) - sequence_length):
    input_sequences.append([note_to_int[n] for n in notes[i:i + sequence_length]])
    output_notes.append(note_to_int[notes[i + sequence_length]])

# Reshape for LSTM
X = np.array(input_sequences)
y = to_categorical(output_notes, num_classes=len(unique_notes))

# Pad sequences to ensure uniform length
X = pad_sequences(X, maxlen=sequence_length, padding='pre')

# ✅ Fix: Define Model BEFORE training starts
checkpoint_path = "models/melody_checkpoint.h5"

if os.path.exists(checkpoint_path):
    print("✅ Resuming training from checkpoint...")
    model = load_model(checkpoint_path)  # ✅ Load existing model
else:
    print("🚀 Starting training from scratch...")

    model = Sequential([
        Embedding(len(unique_notes), 50, input_length=sequence_length),  # Reduce embedding size
        LSTM(128, return_sequences=True),  # Reduce LSTM units
        Dropout(0.2),
        LSTM(128),
        Dense(128, activation='relu'),
        Dense(len(unique_notes), activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Save model after every few epochs
checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="loss")

# Train in small steps (5 epochs at a time)
for i in range(10):  # 10 iterations → 10x5 = 50 epochs
    print(f"\n🔄 Training batch {i+1}/10 (5 epochs)...")
    history = model.fit(X, y, epochs=5, batch_size=16, verbose=1, callbacks=[checkpoint])

# Save Final Model
model.save("models/melody_generator.h5")
print("✅ Full Training Complete! Model saved to models/melody_generator.h5")
