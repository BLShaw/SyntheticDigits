import sys
import os

# Ensure src is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from syntheticdigits.main import main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "generate":
        sys.argv.insert(1, "generate")
    elif len(sys.argv) == 1:
         sys.argv.append("generate")
         
    main()