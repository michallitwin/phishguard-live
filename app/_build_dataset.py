import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.build_dataset import OUTPUT_PATH, build_dataset, save_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

logger.info("Building dataset...")
dataset = build_dataset()
logger.info("Built %d rows", len(dataset))
logger.info("Label counts:\n%s", dataset["label"].value_counts())

save_dataset(dataset)
logger.info("Saved to %s", OUTPUT_PATH)
 