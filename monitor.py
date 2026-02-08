import pyautogui

def movement(x, y, click=False):
    pyautogui.moveTo(x, y)

    if click:
        pyautogui.leftClick(x, y)

def window_size():
    return pyautogui.size()
