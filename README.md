# Spatiotemporal Face Alignment for Generalizable Deepfake Detection (FG 2025)

Official repository for "Spatiotemporal Face Alignment for Generalizable Deepfake Detection" (International Conference on Automatic Face and Gesture Recognition, 2025).

## Setup

### Requirements

Create a python virtual environment (>= 3.10) and run:

```bash
pip install -e .
```

### Datasets

1. Download test video files for each dataset:

    | Dataset name         | Data download                                               |
    |----------------------|-------------------------------------------------------------|
    | CelebDF-v2           | [Link](https://github.com/yuezunli/celeb-deepfakeforensics) |
    | DFDCP                | [Link](https://ai.meta.com/datasets/dfdc)                   |
    | FaceShifter          | [Link](https://github.com/ondyari/FaceForensics)            |
    | DeeperForensics-1.0  | [Link](https://github.com/EndlessSora/DeeperForensics-1.0)  |

    Download bounding box annotation files from the [Release page]().

    Put videos inside a directory called `videos` and extract annotations inside a directory called `annotations`. For example, the directory tree for `FaceShifter` should look like this:

    ```md
    faceforensics/
    ├── annotations/
    │   ├── manipulated_sequences/
    │   └── original_sequences/
    └── videos/
        ├── manipulated_sequences/
        └── original_sequences/
    ```

2. After downloading the required data, run:

    ```bash
    python scripts/configure_data_paths.py [ROOT_PATH]
    ```

    to detect and save all dataset paths inside `ROOT_PATH`.

3. Crop face images from videos, by running:

    ```bash
    python scripts/crop_faces.py [DATASET]
    ```

## Pre-trained weights

Download and extract pre-trained weights from the [Release page]().

## Testing

To compute results for a model, run:

```bash
python scripts/test.py [CONFIG_PATH]
```

`CONFIG_PATH` argument must point to a file named `config.json` inside each experiment directory.

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
