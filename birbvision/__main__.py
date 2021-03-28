from .classifybirb import ClassifyBird
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify an image.')
    parser.add_argument('path', help="Path of the file to classify")
    args = parser.parse_args()

    print(f"Classifying: {args.path}")
    classifier = ClassifyBird()
    matches = classifier.classify_path(args.path)

    for match in matches:
        print(f"{match.label}: {match.confidence}")