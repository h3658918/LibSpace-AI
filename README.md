# LibSpace AI

LibSpace AI is a simple prototype for estimating library occupancy using computer vision.

## Current prototype

The system:
1. Takes a real library image as input.
2. Uses a pretrained YOLOv8s model to detect visible people.
3. Uses a manually configured zone capacity.
4. Estimates occupied seats, available seats and occupancy percentage.
5. Saves an output image showing detected people.

No model training or facial recognition is used.

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt