import os
import shutil
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RunInfo:
    base_name: str  # e.g. "CU"
    prompt_name: str  # e.g. "prompt_4"
    version_num: int  # e.g. 7
    version_dir: str  # correct pv2 output dir
    misplaced_replot: str  # where the replot was actually saved


def _stack_images_vertically_neutral(
    image1_path: str,
    image2_path: str,
    output_path: str,
    border_color=(150, 150, 150),
    border_size: int = 30,
) -> None:
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)
    if img1 is None or img2 is None:
        raise RuntimeError(f"Cannot read images. img1={image1_path} img2={image2_path}")

    width = max(img1.shape[1], img2.shape[1])
    img1_resized = cv2.resize(img1, (width, int(img1.shape[0] * width / img1.shape[1])))
    img2_resized = cv2.resize(img2, (width, int(img2.shape[0] * width / img2.shape[1])))

    label_height = 60
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.8
    font_thickness = 3

    label1 = np.ones((label_height, width, 3), dtype=np.uint8) * 255
    label2 = np.ones((label_height, width, 3), dtype=np.uint8) * 255
    cv2.putText(label1, "Original", (15, 45), font, font_scale, (0, 0, 0), font_thickness)
    cv2.putText(label2, "Extracted (Re-plotted)", (15, 45), font, font_scale, (0, 0, 0), font_thickness)

    combined_image = np.vstack((label1, img1_resized, label2, img2_resized))

    combined_image_with_border = cv2.copyMakeBorder(
        combined_image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=border_color,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, combined_image_with_border)


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    plots_root = os.path.join(repo_root, "plots")

    runs = [
        RunInfo(
            base_name="CU",
            prompt_name="prompt_4",
            version_num=7,
            version_dir=os.path.join(
                plots_root, "synthetic", "C", "CU", "CU_png.pv2_prompt_4.v7"
            ),
            misplaced_replot=os.path.join(
                plots_root,
                "synthetic",
                "C",
                "C",
                "CU",
                "CU_png.pv2_prompt_4.v7",
                "CU_png-replot.pv2_prompt_4.v7.png",
            ),
        ),
        RunInfo(
            base_name="CV",
            prompt_name="prompt_4",
            version_num=3,
            version_dir=os.path.join(
                plots_root, "synthetic", "C", "CV", "CV_png.pv2_prompt_4.v3"
            ),
            misplaced_replot=os.path.join(
                plots_root,
                "synthetic",
                "C",
                "CV_png.pv2_prompt_4.v3",
                "CV_png-replot.pv2_prompt_4.v3.png",
            ),
        ),
        RunInfo(
            base_name="CY",
            prompt_name="prompt_4",
            version_num=1,
            version_dir=os.path.join(
                plots_root, "synthetic", "C", "CY", "CY_png.pv2_prompt_4.v1"
            ),
            misplaced_replot=os.path.join(
                plots_root,
                "synthetic",
                "C",
                "C",
                "CY_png.pv2_prompt_4.v1",
                "CY_png-replot.pv2_prompt_4.v1.png",
            ),
        ),
    ]

    for run in runs:
        original_image = os.path.join(plots_root, "synthetic", "C", run.base_name, f"{run.base_name}.png")
        expected_replot = os.path.join(run.version_dir, f"{run.base_name}_png-replot.pv2_{run.prompt_name}.v{run.version_num}.png")

        # The expected filename should be "<base>_png-replot.pv2_prompt_4.vN.png".
        # If it already exists, don't overwrite.
        if not os.path.exists(expected_replot):
            if not os.path.exists(run.misplaced_replot):
                raise FileNotFoundError(f"Misplaced replot not found: {run.misplaced_replot}")
            os.makedirs(run.version_dir, exist_ok=True)
            shutil.copy2(run.misplaced_replot, expected_replot)

        comparison_out = os.path.join(run.version_dir, f"comparison_{run.base_name}.{run.prompt_name}.v{run.version_num}.png")
        if not os.path.exists(comparison_out):
            _stack_images_vertically_neutral(original_image, expected_replot, comparison_out)

        print(f"[OK] {run.base_name} v{run.version_num}: wrote {os.path.relpath(comparison_out, repo_root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
