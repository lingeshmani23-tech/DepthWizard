"""
Script to create sample benchmark images for DepthWizard demo mode.
"""

import os
import numpy as np
import cv2
from PIL import Image

def generate_sample_person_building(path: str):
    h, w = 512, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Sky gradient
    for y in range(250):
        b = int(255 - y * 0.5)
        g = int(200 - y * 0.3)
        r = int(130 - y * 0.2)
        img[y, :] = [r, g, b]

    # Ground (grass/concrete)
    for y in range(250, h):
        img[y, :] = [60, 120, 60]

    # Building (Target)
    cv2.rectangle(img, (380, 60), (540, 430), (140, 100, 80), -1)
    cv2.rectangle(img, (380, 60), (540, 430), (80, 60, 50), 3)
    # Building Windows
    for wy in range(100, 400, 60):
        for wx in range(410, 520, 40):
            cv2.rectangle(img, (wx, wy), (wx + 25, wy + 40), (220, 240, 255), -1)

    # Reference Person (Left)
    # Head
    cv2.circle(img, (182, 170), 20, (230, 190, 160), -1)
    # Body
    cv2.rectangle(img, (170, 190), (195, 330), (40, 60, 180), -1)
    # Legs
    cv2.line(img, (178, 330), (178, 420), (30, 30, 30), 6)
    cv2.line(img, (188, 330), (188, 420), (30, 30, 30), 6)

    # Label text overlay
    cv2.putText(img, "DepthWizard Benchmark Scene A", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    Image.fromarray(img).save(path)


def generate_sample_street_scene(path: str):
    h, w = 512, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Sky
    img[:300, :] = [200, 180, 140]
    # Ground
    img[300:, :] = [80, 80, 80]

    # Traffic light pole (Target)
    cv2.rectangle(img, (382, 100), (398, 480), (100, 100, 100), -1)
    # Traffic light box
    cv2.rectangle(img, (365, 80), (415, 180), (40, 40, 40), -1)
    cv2.circle(img, (390, 105), 12, (255, 50, 50), -1)
    cv2.circle(img, (390, 130), 12, (255, 200, 50), -1)
    cv2.circle(img, (390, 155), 12, (50, 255, 50), -1)

    # Pedestrian (Reference)
    cv2.circle(img, (122, 220), 18, (220, 180, 150), -1)
    cv2.rectangle(img, (110, 238), (135, 360), (200, 50, 50), -1)
    cv2.line(img, (118, 360), (118, 450), (20, 20, 20), 5)
    cv2.line(img, (127, 360), (127, 450), (20, 20, 20), 5)

    cv2.putText(img, "DepthWizard Benchmark Scene B", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    Image.fromarray(img).save(path)


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    generate_sample_person_building(os.path.join(out_dir, "sample_person_building.jpg"))
    generate_sample_street_scene(os.path.join(out_dir, "sample_street_scene.jpg"))
    print("Sample benchmark images generated successfully.")
