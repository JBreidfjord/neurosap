import os

import cv2

from neurosap.index import food_index, pet_index

# Pets
pet_dir = "images/pets/"
# Load images
pet_imgs = [cv2.imread(pet_dir + p + ".png") for p in pet_index]
# Convert to grayscale
pet_imgs = [cv2.cvtColor(p, cv2.COLOR_BGR2GRAY) for p in pet_imgs]
# Detect edges
pet_imgs = [cv2.Canny(p, 50, 200) for p in pet_imgs]
# Create dict mapping pet names to images
pet_imgs = {pet_index[i]: pet_imgs[i] for i in range(len(pet_index))}

# Foods
food_dir = "images/foods/"
food_imgs = [cv2.imread(food_dir + f + ".png") for f in food_index]
food_imgs = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in food_imgs]
food_imgs = [cv2.Canny(f, 50, 200) for f in food_imgs]
food_imgs = {food_index[i]: food_imgs[i] for i in range(len(food_index))}

# Trophies
trophy_dir = "images/trophies/"
trophy_imgs = {int(f[:-4]): cv2.imread(trophy_dir + f) for f in os.listdir(trophy_dir)}

# Numeric Values
number_dir = "images/numbers/"
# Image names must be converted to int later
# due to having underscores in the name for multiple values
num_imgs = {f[:-4]: cv2.imread(number_dir + f) for f in os.listdir(number_dir)}
num_imgs = {k: cv2.cvtColor(v, cv2.COLOR_BGR2GRAY) for k, v in num_imgs.items()}
num_imgs = {k: cv2.Canny(v, 50, 200) for k, v in num_imgs.items()}

# Attack
attack_dir = "images/attack/"
att_imgs = {int(f[:-4]): cv2.imread(attack_dir + f) for f in os.listdir(attack_dir)}

# Health
health_dir = "images/health/"
hp_imgs = {int(f[:-4]): cv2.imread(health_dir + f) for f in os.listdir(health_dir)}
