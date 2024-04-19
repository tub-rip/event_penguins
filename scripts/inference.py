import sys
import os
import json

from absl import app
from absl import flags
from absl import logging
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils import get_config
from src.proposals import ProposalGenerator
from src.classification import ProposalClassifier
from src.evaluation import DetectionsEvaluator

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "config", "config/exe/sort_tracking/debug.yaml", "Path to the config file"
)
flags.DEFINE_enum(
    "log_level",
    "info",
    ["debug", "info", "warning", "error", "critical"],
    "Set the logging level (default: info)",
)
flags.DEFINE_boolean(
    "verbose", False, "Write output to terminal (True), or log file (False)."
)


def propagate_config(config):
    config["proposals"]["data_path"] = config["common"]["data_path"]
    config["classification"]["data_path"] = config["common"]["data_path"]
    return config


def main(argv):
    del argv
    logging.set_verbosity(FLAGS.log_level.upper())
    root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
    config = get_config(FLAGS.config, root)
    config = propagate_config(config)

    output_dir = config["output_dir"]
    if not FLAGS.verbose:
        logging.get_absl_handler().use_absl_log_file(os.path.join(output_dir, "log"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # 1st stage: proposal generation
    proposal_config = config["proposals"]
    proposal_generator = ProposalGenerator(**proposal_config)
    proposals = proposal_generator.run()

    # 2nd stage: proposal classification
    classifier_config = config["classification"]
    classifier = ProposalClassifier(device, **classifier_config)
    results = classifier.run(proposals)

    # Evaluation
    pred_path = os.path.join(output_dir, "predictions.json")
    with open(pred_path, "w") as f:
        json.dump(results, f, indent=2)

    evaluation_config = config["evaluation"]
    evaluator = DetectionsEvaluator(
        prediction_filename=pred_path,
        valid_labels="ed",
        valid_sequences=list(results["results"].keys()),
        **evaluation_config,
    )
    mean_ap = evaluator.run()

    logging.info(f"mAP: {mean_ap}")
    logging.info("Done.")


if __name__ == "__main__":
    app.run(main)
