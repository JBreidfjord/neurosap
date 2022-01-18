import cv2
import numpy as np

from neurosap.index import encoding_index, food_index, pet_index
from neurosap.images import (
    att_imgs,
    food_imgs,
    hp_imgs,
    num_imgs,
    pet_imgs,
    trophy_imgs,
)


def encode_slot(item: str, x: int, y: int, data: list[int]):
    """Encodes slot for pets and foods"""
    type = None
    valid_x = [310, 405, 500, 600, 695, 790, 885]

    # Check if on team (y = 225)
    if abs(y - 225) < abs(y - 415):
        type = "Team"
        index = 4  # add 6
        valid_x = valid_x[:-2]

    # Check for slot
    # Finds minimum distance from x value
    slot, _ = min(enumerate(valid_x), key=lambda vx: abs(x - vx[1]))

    if slot > 4:
        type = "Food"
        slot -= 5
        index = 49 + slot
    elif type == "Team":
        index += slot * 6
    else:
        type = "Shop"
        index = 34 + slot * 3

    data[index] = food_index.index(item) if type == "Food" else pet_index.index(item)
    return data


def encode_value_slot(n: int, x: int, y: int, data: list[int]):
    """Encodes slot for numeric values"""
    if abs(y - 20) > 10:
        return

    value_types = ["coins", "lives", "turn"]
    valid_x = [60, 170, 430]
    # Finds minimum distance from x value
    nearest, _ = min(zip(value_types, valid_x), key=lambda vx: abs(x - vx[1]))

    index = encoding_index.index(nearest)
    data[index] = n
    return data


def encode_attack_slot(att: int, x: int, y: int, data: list[int]):
    """Encodes slot for attack value"""
    valid_x = [310, 406, 502, 598, 694]
    # Finds minimum distance from x value
    slot, _ = min(enumerate(valid_x), key=lambda vx: abs(x - vx[1]))
    # Distance from y value determines if this is a team or shop pet
    index = 5 + slot * 6 if abs(y - 312) < abs(y - 504) else 35 + slot * 3
    data[index] = att
    return data


def encode_health_slot(hp: int, x: int, y: int, data: list[int]):
    """Encodes slot for health value"""
    valid_x = [356, 452, 548, 644, 740]
    # Finds minimum distance from x value
    slot, _ = min(enumerate(valid_x), key=lambda vx: abs(x - vx[1]))
    # Distance from y value determines if this is a team or shop pet
    index = 6 + slot * 6 if abs(y - 312) < abs(y - 504) else 36 + slot * 3
    data[index] = hp
    return data

# Pets
default_threshold = 0.3
threshold_overrides = {
    "ant": 0.21,
    "badger": 0.18,
    "bison": 0.18,
    "deer": 0.22,
    "dodo": 0.19,
    "dog": 0.2,
    "elephant": 0.28,
    "fish": 0.19,
    "giraffe": 0.15,
    "hippo": 0.2,
    "kangaroo": 0.2,
    "mammoth": 0.23,
    "monkey": 0.13,
    "parrot": 0.22,
    "penguin": 0.25,
    "rat": 0.25,
    "rhino": 0.24,
    "rooster": 0.22,
    "scorpion": 0.14,
    "shark": 0.25,
    "sheep": 0.16,
    "snail": 0.2,
    "snake": 0.2,
    "squirrel": 0.26,
    "swan": 0.2,
    "tiger": 0.24,
    "turtle": 0.17,
}
for pet, pet_img in pet_imgs.items():
    result = cv2.matchTemplate(game_img, pet_img, cv2.TM_CCOEFF_NORMED)
    w = pet_img.shape[1]  # Width
    h = pet_img.shape[0]  # Height
    threshold = threshold_overrides.get(pet, default_threshold)
    loc = np.where(result >= threshold)
    if len(loc[0]) > 0:
        game_rgb = cv2.imread("game.png")
        for pt in zip(*loc[::-1]):
            data = encode_slot(pet, *pt, data)

    if pet in []:  # For finding threshold overrides
        _, max_val, _, _ = cv2.minMaxLoc(result)
        print(pet, max_val)

# Food
threshold = 0.2
for food, food_img in food_imgs.items():
    result = cv2.matchTemplate(game_img, food_img, cv2.TM_CCOEFF_NORMED)
    w = food_img.shape[1]  # Width
    h = food_img.shape[0]  # Height
    loc = np.where(result >= threshold)
    if len(loc[0]) > 0:
        game_rgb = cv2.imread("game.png")
        for pt in zip(*loc[::-1]):
            data = encode_slot(food, *pt, data)

# Trophies
best_val = 0
best = None
for i, img in trophy_imgs.items():
    result = cv2.matchTemplate(game_rgb, img, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    if max_val > best_val:
        best_val = max_val
        best = i
data[2] = best

# Numeric Values
for n, img in num_imgs.items():
    n = int(n.replace("_", ""))
    result = cv2.matchTemplate(game_img, img, cv2.TM_CCOEFF_NORMED)
    h, w = img.shape
    threshold = 0.9
    loc = np.where(result >= threshold)
    for pt in zip(*loc[::-1]):
        data = encode_value_slot(n, *pt, data)

# Attack
for att, img in att_imgs.items():
    result = cv2.matchTemplate(game_rgb, img, cv2.TM_CCOEFF_NORMED)
    h, w, _ = img.shape
    threshold = 0.95
    loc = np.where(result >= threshold)
    for pt in zip(*loc[::-1]):
        data = encode_attack_slot(att, *pt, data)

# Health
for hp, img in hp_imgs.items():
    result = cv2.matchTemplate(game_rgb, img, cv2.TM_CCOEFF_NORMED)
    h, w, _ = img.shape
    threshold = 0.95
    loc = np.where(result >= threshold)
    for pt in zip(*loc[::-1]):
        data = encode_health_slot(hp, *pt, data)


