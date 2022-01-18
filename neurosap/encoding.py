import cv2
import numpy as np

from neurosap.images import (
    att_imgs,
    food_imgs,
    hp_imgs,
    lvl_imgs,
    num_imgs,
    pet_imgs,
    trophy_imgs,
)
from neurosap.index import (
    decoding_index,
    encoding_index,
    food_index,
    pet_index,
    threshold_overrides,
)


def encode(rgb_img, gray_img, data: list[int]):
    data = encode_pets(gray_img, data)
    data = encode_foods(gray_img, data)
    data = encode_trophies(rgb_img, data)
    data = encode_numeric(gray_img, data)
    data = encode_stats(rgb_img, data)
    data = encode_levels(rgb_img, data)

    return data


def encode_pets(img, data: list[int]):
    default_threshold = 0.3
    for pet, pet_img in pet_imgs.items():
        result = cv2.matchTemplate(img, pet_img, cv2.TM_CCOEFF_NORMED)
        threshold = threshold_overrides.get(pet, default_threshold)
        loc = np.where(result >= threshold)
        if len(loc[0]) > 0:
            for pt in zip(*loc[::-1]):
                data = encode_slot(pet, *pt, data)

        if pet in []:  # For finding threshold overrides
            _, max_val, _, _ = cv2.minMaxLoc(result)
            print(pet, max_val)

    return data


def encode_foods(img, data: list[int]):
    threshold = 0.3
    for food, food_img in food_imgs.items():
        result = cv2.matchTemplate(img, food_img, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= threshold)
        if len(loc[0]) > 0:
            for pt in zip(*loc[::-1]):
                data = encode_slot(food, *pt, data)

    return data


def encode_trophies(img, data: list[int]):
    best_val = 0
    best = None
    for i, trophy_img in trophy_imgs.items():
        result = cv2.matchTemplate(img, trophy_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val > best_val:
            best_val = max_val
            best = i
    data[2] = best

    return data


def encode_numeric(img, data: list[int]):
    for n, num_img in num_imgs.items():
        n = int(n.replace("_", ""))
        result = cv2.matchTemplate(img, num_img, cv2.TM_CCOEFF_NORMED)
        threshold = 0.9
        loc = np.where(result >= threshold)
        for pt in zip(*loc[::-1]):
            data = encode_value_slot(n, *pt, data)

    return data


def encode_stats(img, data: list[int]):
    for att, att_img in att_imgs.items():
        result = cv2.matchTemplate(img, att_img, cv2.TM_CCOEFF_NORMED)
        threshold = 0.95
        loc = np.where(result >= threshold)
        for pt in zip(*loc[::-1]):
            data = encode_attack_slot(att, *pt, data)

    for hp, hp_img in hp_imgs.items():
        result = cv2.matchTemplate(img, hp_img, cv2.TM_CCOEFF_NORMED)
        threshold = 0.95
        loc = np.where(result >= threshold)
        for pt in zip(*loc[::-1]):
            data = encode_health_slot(hp, *pt, data)

    return data


def encode_levels(img, data: list[int]):
    for lvl, lvl_img in lvl_imgs.items():
        lvl, exp = lvl.split("_")
        lvl = int(lvl)
        exp = int(exp)

        result = cv2.matchTemplate(img, lvl_img, cv2.TM_CCOEFF_NORMED)
        threshold = 0.95
        loc = np.where(result >= threshold)
        for pt in zip(*loc[::-1]):
            data = encode_level_slot(lvl, exp, pt[0], data)

    return data


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


def encode_level_slot(level: int, exp: int, x: int, data: list[int]):
    """Encodes slot for level and experience values"""
    valid_x = [301, 397, 493, 589, 685]
    # Finds minimum distance from x value
    slot, _ = min(enumerate(valid_x), key=lambda vx: abs(x - vx[1]))
    lvl_idx, exp_idx = (8 + slot * 6, 9 + slot * 6)
    data[lvl_idx] = level
    data[exp_idx] = exp

    return data


def decode(idx: int):
    return decoding_index[idx]
