import os

pet_dir = "images/pets/"
food_dir = "images/foods/"
pet_index = [f[:-4] for f in sorted(os.listdir(pet_dir))]
food_index = [f[:-4] for f in sorted(os.listdir(food_dir))]

