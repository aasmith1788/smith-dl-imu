import argparse

def makeArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        type=str,
        help="type your model name",
    )
    return parser.parse_args()
