import sys
from .main import call_me_maybe

if __name__ == "__main__":
    try:
        call_me_maybe()
    except KeyboardInterrupt:
        print("User interrupted the program",
              file=sys.stderr)
        sys.exit(1)
