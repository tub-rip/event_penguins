import sys
import os
import json

from absl import app
from absl import flags

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation import DetectionsEvaluator

FLAGS = flags.FLAGS
flags.DEFINE_string("prediction_path", None, "Path to prediction file.")


def propagate_config(config):
    config["proposals"]["data_dir"] = config["common"]["data_dir"]
    config["classification"]["data_dir"] = config["common"]["data_dir"]
    return config


def main(argv):
    del argv

    with open(FLAGS.prediction_path, "r") as file:
        data = json.load(file)

    evaluator = DetectionsEvaluator(
        ground_truth_filename="config/annotations/annotations.json",
        prediction_filename="predictions.json",
        valid_labels="ed",
        tiou_thresholds=[0.1, 0.3, 0.5, 0.7],
        valid_sequences=list(data["results"].keys()),
    )
    mean_ap = evaluator.run()

    print(f"mAP: {mean_ap}")
    print("Done.")


if __name__ == "__main__":
    app.run(main)
