import cv2
import pyautogui

from neurosap.index import decoding_index, encoding_index, food_index, pet_index


def get_game_imgs():
    """
    Screenshots game window.\n
    Returns 2-tuple of RGB image and grayscale image
    """
    pyautogui.screenshot("temp/game.png", region=(640, 371, 1280, 720))
    game_rgb = cv2.imread("temp/game.png")
    game_img = cv2.cvtColor(game_rgb, cv2.COLOR_BGR2GRAY)
    game_img = cv2.Canny(game_img, 50, 200)

    # Additional screenshots
    pyautogui.screenshot("temp/coin.png", region=(700, 392, 40, 40))
    pyautogui.screenshot("temp/turn.png", region=(1065, 392, 40, 40))
    pyautogui.screenshot("temp/trophy.png", region=(913, 392, 65, 40))
    pyautogui.screenshot("temp/lives.png", region=(810, 392, 40, 40))

    pyautogui.screenshot("temp/t0_att.png", region=(950, 683, 40, 40))
    pyautogui.screenshot("temp/t0_hp.png", region=(996, 683, 40, 40))

    pyautogui.screenshot("temp/s0_att.png", region=(950, 873, 40, 40))
    pyautogui.screenshot("temp/s0_hp.png", region=(996, 873, 40, 40))

    pyautogui.screenshot("3_0.png", region=(941, 561, 40, 40))

    return game_rgb, game_img


def is_game_over():
    rgb_img, _ = get_game_imgs()
    img = cv2.imread("images/start.png")
    result = cv2.matchTemplate(rgb_img, img, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return max_val >= 0.95 and abs(max_loc[0] - 260) <= 10 and abs(max_loc[1] - 154) <= 10


def get_legal_moves(data: list[int]):
    legal = [False for _ in range(len(decoding_index))]

    if data[0] == 0:  # No coins
        for i in range(20, 55):
            legal[i] = True
        legal[60] = True  # No roll
    elif data[0] < 3:  # Not enough coins to buy
        for i in range(20, 45):  # Shop pets
            legal[i] = True
        if food_index[data[50]] != "sleeping_pill":  # Sleeping pill costs 1
            for i in range(45, 50):  # Shop foods
                legal[i] = True
        if food_index[data[51]] != "sleeping_pill":
            for i in range(50, 55):  # Shop foods
                legal[i] = True

    if (  # All team slots full or no shop pets remaining
        data[4] != -1 and data[10] != -1 and data[16] != -1 and data[22] != -1 and data[28] != -1
    ) or (
        data[34] == -1 and data[37] == -1 and data[40] == -1 and data[43] == -1 and data[46] == -1
    ):
        for i in range(20, 45):  # Shop pets
            legal[i] = True

    for i, j in enumerate(range(4, 29, 6)):  # Team pet ids
        if data[j] == -1:  # Can't interact with pets that don't exist
            for k in range(0 + i * 4, 4 + i * 4):  # No move
                legal[k] = True
            legal[45 + i] = True  # No food
            legal[50 + i] = True  # No food
            legal[55 + i] = True  # No sell

    for i, j in enumerate(range(34, 49, 3)):  # Shop pet ids
        if data[j] == -1:  # Can't interact with pets that don't exist
            for k in range(20 + i * 5, 25 + i * 5):
                legal[k] = True

    for i, j in enumerate([49, 50]):  # Food ids
        if data[j] == -1:
            for k in range(45 + i * 5, 50 + i * 5):
                legal[k] = True

    return legal


def display_data(data: list[int]):
    out = "Info:\n"
    team_id_indices = [4, 10, 16, 22, 28]
    team_att_indices = [5, 11, 17, 23, 29]
    team_hp_indices = [6, 12, 18, 24, 30]
    team_item_indices = [7, 13, 19, 25, 31]
    team_lvl_indices = [8, 14, 20, 26, 32]
    team_exp_indices = [9, 15, 21, 27, 33]
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
            team[idx] = "[" + pet_index[x]
        elif i in team_att_indices:
            idx = team_att_indices.index(i)
            team[idx] += f" ({x}|"
        elif i in team_hp_indices:
            idx = team_hp_indices.index(i)
            team[idx] += f"{x})"
        elif i in team_item_indices:
            idx = team_item_indices.index(i)
            team[idx] += f"(i{x})"
        elif i in team_lvl_indices:
            idx = team_lvl_indices.index(i)
            team[idx] += f"({x}|"
        elif i in team_exp_indices:
            idx = team_exp_indices.index(i)
            team[idx] += f"{x})]"
        elif i in shop_id_indices:
            idx = shop_id_indices.index(i)
            shop[idx] = "[" + pet_index[x]
        elif i in shop_att_indices:
            idx = shop_att_indices.index(i)
            shop[idx] += f" ({x}|"
        elif i in shop_hp_indices:
            idx = shop_hp_indices.index(i)
            shop[idx] += f"{x})]"
        elif i in food_id_indices:
            idx = food_id_indices.index(i)
            food[idx] = "[" + food_index[x] + "]"
        elif i == 0:
            out += f"{encoding_index[i].title()}: {x}"
        else:
            out += f" | {encoding_index[i].title()}: {x}"

    team = [t if t else "[]" for t in team]
    shop = [s if s else "[]" for s in shop]
    food = [f if f else "[]" for f in food]
    out += "\n" + " ".join(team)
    out += "\n" + " ".join(shop)
    out += "\n" + " ".join(food)

    return out
