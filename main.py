from build_dataset import build_dataset
from train_classifier import train_classifier
from pose_finder import pose_finder

def main():
    build_dataset()
    train_classifier()
    pose_finder()
    
if __name__ == "__main__":
    main()