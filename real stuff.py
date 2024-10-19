import os
import numpy as np
from PIL import Image
from collections import defaultdict
import random

# Load the dataset of images to learn pixel transitions
def load_image_data(folder_path, image_size=(1024, 1024)):
    data = []
    for i in range(1, 7):  # Load 6 images (1.png to 6.png)
        img_path = os.path.join(folder_path, f'{i}.png')
        img = Image.open(img_path).resize(image_size)
        img_data = np.array(img)
        data.append(img_data)
    return data

# Function to extract both pixel transitions and their position in the image
def extract_pixel_transitions_with_position(image_data):
    transition_counts = defaultdict(lambda: defaultdict(int))
    position_color_distribution = defaultdict(lambda: defaultdict(int))
    
    img_height, img_width = image_data[0].shape[:2]  # Assume all images have the same size
    
    for img in image_data:
        for i in range(img_height):
            for j in range(img_width):
                current_pixel = tuple(img[i, j])  # Current pixel value (R, G, B)
                
                # Track color distribution at each position (i, j)
                position_color_distribution[(i, j)][current_pixel] += 1
                
                # Transition right
                if j + 1 < img_width:
                    right_neighbor = tuple(img[i, j + 1])
                    transition_counts[(current_pixel, (i, j))][(right_neighbor, (i, j + 1))] += 1
                
                # Transition down
                if i + 1 < img_height:
                    bottom_neighbor = tuple(img[i + 1, j])
                    transition_counts[(current_pixel, (i, j))][(bottom_neighbor, (i + 1, j))] += 1
    
    return transition_counts, position_color_distribution

# Normalize the probabilities for both transitions and positional color distribution
def normalize_probabilities(transition_counts, position_color_distribution):
    transition_probs = {}
    position_probs = {}
    
    # Normalize transition counts to transition probabilities
    for (current_pixel, pos), next_pixel_dict in transition_counts.items():
        total_transitions = sum(next_pixel_dict.values())
        transition_probs[(current_pixel, pos)] = {next_pixel: count / total_transitions 
                                                 for next_pixel, count in next_pixel_dict.items()}
    
    # Normalize position-specific color distributions
    for pos, color_dict in position_color_distribution.items():
        total_colors = sum(color_dict.values())
        position_probs[pos] = {color: count / total_colors for color, count in color_dict.items()}
    
    return transition_probs, position_probs

# Train the HMM to create the hidden state (generator algorithm) based on images
def train_hmm(images):
    transition_counts, position_color_distribution = extract_pixel_transitions_with_position(images)
    transition_probs, position_probs = normalize_probabilities(transition_counts, position_color_distribution)
    return transition_probs, position_probs

# Function to seed the randomness based on seed value (to get deterministic outputs for the same seed)
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# Function to generate a new image based on seed and text input using the trained HMM
def generate_image(generator_model, seed, image_size=(1024, 1024)):
    transition_probs, position_probs = generator_model
    set_random_seed(seed)  # Ensure the seed controls randomness
    generated_img = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

    img_height, img_width = image_size
    
    # Start with the first pixel based on the learned position probability
    for i in range(img_height):
        for j in range(img_width):
            pos = (i, j)
            
            # First check the most likely color for this position
            if pos in position_probs:
                possible_colors = position_probs[pos]
                color = random.choices(list(possible_colors.keys()), list(possible_colors.values()))[0]
                generated_img[i, j] = color
            
            # If we have a pixel to the left or above, use transition probabilities
            if i > 0 or j > 0:
                # Try to predict the next pixel based on transitions from neighbors
                current_pixel = tuple(generated_img[i, j - 1] if j > 0 else generated_img[i - 1, 0])
                if (current_pixel, pos) in transition_probs:
                    next_pixel = random.choices(
                        list(transition_probs[(current_pixel, pos)].keys()), 
                        list(transition_probs[(current_pixel, pos)].values())
                    )[0][0]
                    generated_img[i, j] = next_pixel
                else:
                    # Fallback: random pixel from learned position probabilities
                    color = random.choices(list(possible_colors.keys()), list(possible_colors.values()))[0]
                    generated_img[i, j] = color
    
    return Image.fromarray(generated_img)

# Main function
def main():
    folder_path = 'flower'  # The folder containing the images
    image_size = (1024, 1024)  # Image dimensions

    # Step 1: Load the images to learn pixel transitions
    images = load_image_data(folder_path, image_size)

    # Step 2: Train the HMM generator model on the images
    generator_model = train_hmm(images)

    # Step 3: Generate a new image based on a random seed
    new_seed = random.randint(7, 1000)  # Random new seed not in the training set
    generated_image = generate_image(generator_model, new_seed, image_size)
    generated_image.save(f"OUT_seed_{new_seed}.png")
    print(f"Generated new image with seed {new_seed} saved as 'OUT_seed_{new_seed}.png'")

if __name__ == "__main__":
    main()
