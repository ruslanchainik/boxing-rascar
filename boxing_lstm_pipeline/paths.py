from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_VIDEOS = ROOT / "train" / "videos.csv"
TRAIN_PUNCHES = ROOT / "train" / "punches.csv"
TEST_VIDEOS = ROOT / "test" / "videos.csv"
SAMPLE_SUBMISSION = ROOT / "sample_submission.csv"
ARTIFACTS = ROOT / "artifacts"
POSE_DIR = ARTIFACTS / "pose_features"
MODEL_DIR = ARTIFACTS / "models"

