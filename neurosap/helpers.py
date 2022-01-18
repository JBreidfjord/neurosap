import cv2
import pyautogui

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