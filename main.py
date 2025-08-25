"""
Unused for now, but will be an entry point for eval
and API later.
"""

import logging


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s][%(name)s]: %(message)s",
    )
    print("Oh :(. I'm not ready yet!")


if __name__ == "__main__":
    main()
