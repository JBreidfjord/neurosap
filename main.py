import pyautogui
import cv2
import numpy as np
from encoding import encoding_index, pet_index, food_index
from images import pet_imgs, food_imgs, trophy_imgs, num_imgs, att_imgs, hp_imgs


def get_slot(item: str, x: int, y: int, data: list[int]):
    """Gets slot for pets and foods"""
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


def get_value_slot(n: int, x: int, y: int, data: list[int]):
    """Gets slot for numeric values"""
    if abs(y - 20) > 10:
        return

    value_types = ["coins", "lives", "turn"]
    valid_x = [60, 170, 430]
    # Finds minimum distance from x value
    nearest, _ = min(zip(value_types, valid_x), key=lambda vx: abs(x - vx[1]))

    index = encoding_index.index(nearest)
    data[index] = n
    return data


def get_attack_slot(att: int, x: int, y: int, data: list[int]):
    """Gets slot for attack value"""
    valid_x = [310, 406, 502, 598, 694]
    # Finds minimum distance from x value
    slot, _ = min(enumerate(valid_x), key=lambda vx: abs(x - vx[1]))
    # Distance from y value determines if this is a team or shop pet
    index = 5 + slot * 6 if abs(y - 312) < abs(y - 504) else 35 + slot * 3
    data[index] = att
    return data
    

def get_health_slot(hp: int, x: int, y: int, data: list[int]):
    """Gets slot for health value"""
    valid_x = [356, 452, 548, 644, 740]
    # Finds minimum distance from x value
    slot, _ = min(enumerate(valid_x), key=lambda vx: abs(x - vx[1]))
    # Distance from y value determines if this is a team or shop pet
    index = 6 + slot * 6 if abs(y - 312) < abs(y - 504) else 36 + slot * 3
    data[index] = hp
    return data
    


# print(pyautogui.position())
pyautogui.screenshot("game.png", region=(640, 371, 1280, 720))
game_rgb = cv2.imread("game.png")
game_img = cv2.cvtColor(game_rgb, cv2.COLOR_BGR2GRAY)
game_img = cv2.Canny(game_img, 50, 200)

coin_img = pyautogui.screenshot("coin.png", region=(700, 392, 40, 40))
turn_img = pyautogui.screenshot("turn.png", region=(1065, 392, 40, 40))
trophy_img = pyautogui.screenshot("trophy.png", region=(913, 392, 65, 40))
lives_img = pyautogui.screenshot("lives.png", region=(810, 392, 40, 40))

t0_att_img = pyautogui.screenshot("t0_att.png", region=(950, 683, 40, 40))
t0_hp_img = pyautogui.screenshot("t0_hp.png", region=(996, 683, 40, 40))

s0_att_img = pyautogui.screenshot("s0_att.png", region=(950, 873, 40, 40))
s0_hp_img = pyautogui.screenshot("s0_hp.png", region=(996, 873, 40, 40))
# s1_att_img = pyautogui.screenshot("s1_att.png", region=(1045, 873, 40, 40))
# s1_hp_img = pyautogui.screenshot("s1_hp.png", region=(1091, 873, 40, 40))
# s2_att_img = pyautogui.screenshot("s2_att.png", region=(1140, 873, 40, 40))
# s2_hp_img = pyautogui.screenshot("s2_hp.png", region=(1186, 873, 40, 40))
# s3_att_img = pyautogui.screenshot("s3_att.png", region=(1235, 873, 40, 40))
# s3_hp_img = pyautogui.screenshot("s3_hp.png", region=(1281, 873, 40, 40))
# s4_att_img = pyautogui.screenshot("s4_att.png", region=(1330, 873, 40, 40))
# s4_hp_img = pyautogui.screenshot("s4_hp.png", region=(1376, 873, 40, 40))

# Encoding
# 51 bits
data = [-1 for _ in range(52)]

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
            data = get_slot(pet, *pt, data)

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
            data = get_slot(food, *pt, data)

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
        data = get_value_slot(n, *pt, data)
        
# Attack
for att, img in att_imgs.items():
    result = cv2.matchTemplate(game_rgb, img, cv2.TM_CCOEFF_NORMED)
    h, w, _ = img.shape
    threshold = 0.95
    loc = np.where(result >= threshold)
    for pt in zip(*loc[::-1]):
        data = get_attack_slot(att, *pt, data)

# Health
for hp, img in hp_imgs.items():
    result = cv2.matchTemplate(game_rgb, img, cv2.TM_CCOEFF_NORMED)
    h, w, _ = img.shape
    threshold = 0.95
    loc = np.where(result >= threshold)
    for pt in zip(*loc[::-1]):
        data = get_health_slot(hp, *pt, data)

team_id_indices = [4, 10, 16, 22, 28]
team_att_indices = [5, 11, 17, 23, 29]
team_hp_indices = [6, 12, 18, 24, 30]
shop_id_indices = [34, 37, 40, 43, 46]
shop_att_indices = [35, 38, 41, 44, 47]
shop_hp_indices = [36, 39, 42, 45, 48]
food_id_indices = [49, 50]
team = ["", "", "", "", ""]
shop = ["", "", "", "", ""]
food = ["", ""]
for i, x in enumerate(data):
    if x == -1:
        continue

    if i in team_id_indices:
        idx = team_id_indices.index(i)
        team[idx] = pet_index[x]
    elif i in team_att_indices:
        idx = team_att_indices.index(i)
        team[idx] += f" ({x},"
    elif i in team_hp_indices:
        idx = team_hp_indices.index(i)
        team[idx] += f"{x})"
    elif i in shop_id_indices:
        idx = shop_id_indices.index(i)
        shop[idx] = pet_index[x]
    elif i in shop_att_indices:
        idx = shop_att_indices.index(i)
        shop[idx] += f" ({x},"
    elif i in shop_hp_indices:
        idx = shop_hp_indices.index(i)
        shop[idx] += f"{x})"
    elif i in food_id_indices:
        idx = food_id_indices.index(i)
        food[idx] = food_index[x]
    else:
        print(f"{encoding_index[i]}: {x}")
print(team, shop, food, sep="\n")

# Remaining:
# Attack / Health (more pics)
# Fix trophies (more pics)
# Item (handled when purchased)
# Level / Exp (handled when upgraded)
# Test all pets for threshold overrides
# Identify finish screens and count trophies