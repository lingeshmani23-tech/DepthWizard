"""
DEPTHWIZARD — Quantitative Evaluation Module
Evaluates reference-assisted height estimation accuracy across test benchmark images.
Saves metrics into evaluation/results.csv.
"""

import sys
import csv
from pathlib import Path
import numpy as np

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from scripts.height_estimation import estimate_height

def run_evaluation(output_csv_path=None):
    """
    Evaluates test benchmark images and calculates absolute error and error percentage.
    """
    if output_csv_path is None:
        output_csv_path = config.EVALUATION_DIR / "results.csv"
    else:
        output_csv_path = Path(output_csv_path)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Benchmark test dataset entries (Outdoor real-world samples with known reference and target ground truth)
    benchmark_samples = [
        {
            "image": "test.jpg",
            "description": "Person standing near door/building",
            "ref_top": (200, 150),
            "ref_bot": (200, 380),
            "ref_real_h": 1.70,  # Known person height in meters
            "target_top": (450, 100),
            "target_bot": (450, 480),
            "actual_target_h": 2.45  # Ground truth door height in meters
        },
        {
            "image": "sample_street_scene.jpg",
            "description": "Traffic pole and bus stop shelter",
            "ref_top": (100, 120),
            "ref_bot": (100, 400),
            "ref_real_h": 3.00,  # Known lamp pole height in meters
            "target_top": (320, 180),
            "target_bot": (320, 420),
            "actual_target_h": 2.60  # Bus stop shelter height in meters
        }
    ]

    results = []

    print("=== DEPTHWIZARD QUANTITATIVE EVALUATION ===")
    for sample in benchmark_samples:
        img_name = sample["image"]
        img_path = config.INPUT_DIR / img_name
        if not img_path.exists():
            img_path = config.BASE_DIR / "sample_images" / img_name

        if not img_path.exists():
            print(f"Skipping {img_name}: file not found.")
            continue

        depth_path = config.OUTPUT_DIR / "depth.npy"

        try:
            res = estimate_height(
                image_path=img_path,
                depth_npy_path=depth_path,
                reference_top=sample["ref_top"],
                reference_bottom=sample["ref_bot"],
                reference_real_height_m=sample["ref_real_h"],
                target_top=sample["target_top"],
                target_bottom=sample["target_bot"]
            )

            est_h = res["estimated_target_height_m"]
            act_h = sample["actual_target_h"]

            abs_err = abs(act_h - est_h)
            err_pct = (abs_err / act_h) * 100.0

            row = {
                "Image": img_name,
                "Description": sample["description"],
                "Reference Object": "Known Reference",
                "Reference Height (m)": sample["ref_real_h"],
                "Actual Target Height (m)": act_h,
                "Estimated Height (m)": round(est_h, 3),
                "Absolute Error (m)": round(abs_err, 3),
                "Error %": round(err_pct, 2)
            }
            results.append(row)

            print(f"Image: {img_name} | Actual: {act_h:.2f}m | Est: {est_h:.2f}m | Abs Err: {abs_err:.3f}m | Error: {err_pct:.2f}%")

        except Exception as e:
            print(f"Error evaluating {img_name}: {e}")

    # Write CSV
    if results:
        headers = list(results[0].keys())
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved evaluation results to {output_csv_path.name}")
    else:
        print("No evaluation samples were processed.")

    return results

if __name__ == "__main__":
    run_evaluation()
