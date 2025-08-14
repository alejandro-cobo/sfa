# Spatiotemporal Face Alignment for Generalizable Deepfake Detection (FG 2025)

Official repository for ["Spatiotemporal Face Alignment for Generalizable Deepfake Detection" (International Conference on Automatic Face and Gesture Recognition, 2025)](https://ieeexplore.ieee.org/document/11099346).

## Setup

### Requirements

Create a python virtual environment (>= 3.10) and install dependencies:

```bash
pip install -e .
```

### Datasets

1. Download and extract bounding box annotations from the [Releases page](https://github.com/alejandro-cobo/sfa/releases/tag/v1.0.0).

2. Download test videos for each dataset:

    | Dataset name        | Data download                                               |
    |---------------------|-------------------------------------------------------------|
    | CelebDF-v2          | [Link](https://github.com/yuezunli/celeb-deepfakeforensics) |
    | DFDCP               | [Link](https://ai.meta.com/datasets/dfdc)                   |
    | FaceShifter         | [Link](https://github.com/ondyari/FaceForensics)            |
    | DeeperForensics-1.0 | [Link](https://github.com/EndlessSora/DeeperForensics-1.0)  |

    and move them to a directory called `videos` inside their corresponding dataset. For example, the directory tree for `FaceShifter` should look like this:

    ```md
    faceshifter/
    ├── .dfd-db-id
    ├── annotations/
    │   ├── manipulated_sequences/
    │   └── original_sequences/
    └── videos/
        ├── manipulated_sequences/
        └── original_sequences/
    ```

3. After downloading the required data, run:

    ```bash
    python scripts/configure_data_paths.py [ROOT_PATH]
    ```

    to detect and save all dataset paths inside `ROOT_PATH`.

4. Crop face images from videos, by running:

    ```bash
    python scripts/crop_faces.py [DATASET]
    ```

    Video files are no longer needed after this step.

## Pre-trained weights

Download and extract pre-trained weights from the [Releases page](https://github.com/alejandro-cobo/sfa/releases/tag/v1.0.0).

## Testing

To compute results for a model, run:

```bash
python scripts/test.py [CONFIG_PATH]
```

`CONFIG_PATH` argument must point to a file named `config.json` inside any experiment directory.

## Results

Video-level AUC (%) metrics:

| Method    | CDF       | DFDCP     | FSh       | DFo       | Avg.      |
|-----------|-----------|-----------|-----------|-----------|-----------|
| Baseline  | 87.17     | 78.45     | 99.55     | 98.50     | 90.92     |
| SFA (M=1) | **89.70** | 79.84     | 99.82     | 99.17     | 92.13     |
| SFA (M=2) | 89.52     | **80.58** | **99.84** | **99.24** | **92.30** |

## Citation

If your find our work useful, please consider citing it in your research:

```bibtex
@inproceedings{11099346,
  author={Cobo, Alejandro and Valle, Roberto and Buenaposada, José M. and Baumela, Luis},
  booktitle={2025 IEEE 19th International Conference on Automatic Face and Gesture Recognition (FG)},
  title={Spatiotemporal Face Alignment for Generalizable Deepfake Detection},
  year={2025},
  pages={1-6}
}
```
