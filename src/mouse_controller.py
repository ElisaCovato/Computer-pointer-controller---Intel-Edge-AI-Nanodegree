import pyautogui


class MouseController:
    '''
    This is a class to use to control the mouse pointer.
    It uses the pyautogui library.
    Precision for mouse movement (how much the mouse moves)
    and the speed (how fast it moves) can be set by changing
    precision_dict and speed_dict.
    '''

    def __init__(self, precision, speed):
        precision_dict = {'high': 100, 'low': 1000, 'medium': 500}
        speed_dict = {'fast': 0.1, 'slow': 1, 'medium': 0.5, 'immediate': 0}

        self.precision = precision_dict[precision]
        self.speed = speed_dict[speed]

        self.xc = pyautogui.size()[0] / 2
        self.yc = pyautogui.size()[1] / 2

        # Start pointer from center screen
        pyautogui.PAUSE = 0
        pyautogui.moveTo(self.xc, self.yc)


    def move(self, x, y):
        current_pos = pyautogui.position() # current mouse position on screen
        rel_x = x * self.precision # relative movement on x axis
        rel_y = -1 * y * self.precision # relative movement on y axis
        new_pos = (current_pos[0] + rel_x, current_pos[1] + rel_y ) # new potential position

        # If the new position is still inside the screen, move the pointer
        # otherwise leave the pointer where it is

        if pyautogui.onScreen(new_pos):
            pyautogui.moveRel(rel_x, rel_y, duration=self.speed)
