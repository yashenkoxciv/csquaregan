import time
from termcolor import colored

DEBUG = True

def info(*msg):
    s = '[I] [' + time.ctime() + '] ' + ' '.join(list(map(str, msg)))
    print(colored(s, "white"))


def debug(*msg):
    if DEBUG:
        s = '[D] [' + time.ctime() + '] ' + " ".join(list(map(str, msg)))
        print(colored(s, "magenta"))


def error(*msg):
    s = '[E] [' + time.ctime() + '] ' + " ".join(list(map(str, msg)))
    print(colored(s, "red"))


def success(*msg):
    s = '[+] [' + time.ctime() + '] ' + " ".join(list(map(str, msg)))
    print(colored(s, "green"))


def fail(*msg):
    s = '[X] [' + time.ctime() + '] ' + " ".join(list(map(str, msg)))
    print(colored(s, "red", attrs=['bold']))


def warning(*msg):
    s = '[W] [' + time.ctime() + '] ' + " ".join(list(map(str, msg)))
    print(colored(s, "yellow")) # , attrs=['reverse']
