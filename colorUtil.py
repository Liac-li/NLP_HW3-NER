class COLOR:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    DEFAULT = '\033[39m'


color_support = True


def set_color(s, color):
    if color_support:
        return f"{color}{s}{COLOR.DEFAULT}"
    return f"{s}"  # pragma: no cover
