"""
Accuracy Evaluation Module
Calculates height estimation error metrics (Absolute Error, Percentage Error) and persists CSV records.
"""

import csv
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List


@dataclass
class EvaluationResult:
    image_name: str
    reference_height_m: float
    estimated_height_m: float
    known_height_m: float
    absolute_error_m: float
    percentage_error: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_height(
    image_name: str,
    estimated_height_m: float,
    known_height_m: float,
    reference_height_m: float
) -> EvaluationResult:
    """
    Calculate absolute error and percentage error against ground truth target height.
    """
    if known_height_m <= 0:
        raise ValueError(f"Known height must be > 0. Got {known_height_m}")

    abs_error = abs(estimated_height_m - known_height_m)
    pct_error = (abs_error / known_height_m) * 100.0

    return EvaluationResult(
        image_name=image_name,
        reference_height_m=round(reference_height_m, 2),
        estimated_height_m=round(estimated_height_m, 2),
        known_height_m=round(known_height_m, 2),
        absolute_error_m=round(abs_error, 2),
        percentage_error=round(pct_error, 2)
    )


def save_evaluation_csv(results: List[EvaluationResult], csv_path: str) -> str:
    """
    Append or write evaluation summary table to CSV file.
    Columns: image_name, reference_height_m, estimated_height_m, known_height_m, absolute_error_m, percentage_error
    """
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    fieldnames = ["image_name", "reference_height_m", "estimated_height_m", "known_height_m", "absolute_error_m", "percentage_error"]

    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow(r.to_dict())

    return csv_path
