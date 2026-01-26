import sys
from .startproject import startproject

def main():
    if len(sys.argv) < 3 or sys.argv[1] != "startproject":
        print("Usage: gptmed startproject <projectname>")
        sys.exit(1)
    startproject(sys.argv[2])
