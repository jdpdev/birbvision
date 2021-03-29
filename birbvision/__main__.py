from .classifybirb import ClassifyBird
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify an image.')
    parser.add_argument('path', help="Path of the file to classify")
    args = parser.parse_args()

    print(f"Classifying: {args.path}")
    classifier = ClassifyBird()
    matches = classifier.classify_path(args.path)
    top = matches.get_top_results(5)

    for match in top:
        print(f"{match.label}: {match.confidence} ({match.confidenceDelta})")