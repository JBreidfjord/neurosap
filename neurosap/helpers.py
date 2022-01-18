import cv2
import pyautogui

from neurosap.index import encoding_index, food_index, pet_index


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

    return game_rgb, game_img


def display_data(data: list[int]):
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
