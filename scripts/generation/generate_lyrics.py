import os
# Disable GPU to avoid CUDA errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Load Tokenizer and Create Reverse Mapping
# ------------------------------
with open("models/tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# Create reverse mapping: index -> word for fast lookup
reverse_word_index = {index: word for word, index in tokenizer.word_index.items()}

# ------------------------------
# Load Trained Model
# ------------------------------
model = tf.keras.models.load_model("models/lyrics_generator_model.h5", compile=False)
print("✅ Lyrics Generator Model Loaded Successfully!")

# ------------------------------
# Sampling Functions
# ------------------------------
def top_k_sampling(predictions, k=5, temperature=1.0):
    """
    Sample an index from the top k predictions after applying temperature scaling.
    """
    predictions = np.asarray(predictions).astype("float64")
    # Get indices of top k probabilities
    top_k_indices = predictions.argsort()[-k:][::-1]
    top_k_probs = predictions[top_k_indices]
    
    # Apply temperature scaling
    top_k_probs = np.log(top_k_probs + 1e-8) / temperature
    top_k_probs = np.exp(top_k_probs)
    top_k_probs = top_k_probs / np.sum(top_k_probs)
    
    # Randomly choose one index from the top k using the adjusted probabilities
    return np.random.choice(top_k_indices, p=top_k_probs)

# ------------------------------
# Generate Lyrics Function
# ------------------------------
def generate_lyrics(seed_text, next_words=50, temperature=1.2, top_k=5, max_repetition=3, resample_attempts=5):
    """
    Generate lyrics based on a seed_text.
    
    Args:
        seed_text (str): The starting phrase.
        next_words (int): Number of words to generate.
        temperature (float): Controls randomness (1.0 is baseline).
        top_k (int): Use the top_k most likely words for sampling.
        max_repetition (int): Maximum number of times a word can repeat consecutively.
        resample_attempts (int): Maximum number of attempts to resample if repetition is too high.
    
    Returns:
        str: The generated lyrics.
    """
    generated_words = []
    # Split the seed text to have a list for repetition checking.
    current_text = seed_text.strip()
    
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([current_text])[0]
        token_list = pad_sequences([token_list], maxlen=model.input_shape[1], padding='pre')
        
        predictions = model.predict(token_list, verbose=0)[0]
        # Use top_k sampling with temperature scaling
        predicted_index = top_k_sampling(predictions, k=top_k, temperature=temperature)
        
        predicted_word = reverse_word_index.get(predicted_index, None)
        if predicted_word is None:
            break
        
        # Resample if the predicted word is repeating too many times
        attempts = 0
        while (len(generated_words) >= max_repetition and 
               all(w == predicted_word for w in generated_words[-max_repetition:]) and
               attempts < resample_attempts):
            predicted_index = top_k_sampling(predictions, k=top_k, temperature=temperature * 1.1)
            predicted_word = reverse_word_index.get(predicted_index, None)
            attempts += 1
        
        generated_words.append(predicted_word)
        current_text += " " + predicted_word
    
    return current_text

# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    user_input = input("\n🎤 Enter a starting line for lyrics: ")
    generated_lyrics = generate_lyrics(user_input, next_words=300, temperature=1.2, top_k=5, max_repetition=3, resample_attempts=5)
    print("\n🎶 Generated Lyrics:\n")
    print(generated_lyrics)
    
    # ✅ Save generated lyrics to a file
    with open("output/generated_lyrics.txt", "w", encoding="utf-8") as f:
        f.write(generated_lyrics)
    
    print("\n✅ Generated lyrics saved to output/generated_lyrics.txt")

