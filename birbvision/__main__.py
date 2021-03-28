from .classifybirb import load_and_classify
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify an image.')
    parser.add_argument('path', help="Path of the file to classify")
    args = parser.parse_args()

    print(f"Classifying: {args.path}")
    matches = load_and_classify(args.path)

    for match in matches:
        print(f"{match[0]}: {match[1]}")