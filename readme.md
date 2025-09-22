## MagShield

Official code for our ICCV25 [paper](https://arxiv.org/abs/2506.22907) “MagShield: Towards Better Robustness in Sparse Inertial Motion Capture Under Magnetic Disturbances”. See [project page](https://yz-shiao.github.io/MagShield).

## Usage

### Install dependencies

Our algorithm requires minimal dependencies: `Python 3.8` (strictly required, as ESKF relies on pybind11 compilation) and `PyTorch 1.13.1` (our tested version).

For full integration with existing inertial posers (`PNP` and `DynaIP`), you'll need to install additional dependencies. We recommend you configure the environment dependencies in the following order:

1. Create a Python 3.8 environment
2. Configure [PNP](https://github.com/Xinyu-Yi/PNP) (most complex setup)
3. Configure [DynaIP](https://github.com/dx118/dynaip)

### Prepare SMPL body model

Download SMPL model from [here](https://smpl.is.tue.mpg.de). You should download the version 1.0.0 for Python 2.7 (10 shape PCs). Rename and put the male model file into `models/SMPL_male.pkl`.

### Prepare network weights of inertial posers

1. Download PNP weights from [here](https://github.com/Xinyu-Yi/PNP/raw/page/files/weights.pt), put it into `inertial_poser/PNP/weights/weights.pt`
2. Download DynaIP weights from [here](https://github.com/dx118/dynaip/blob/main/weights/DynaIP_s.pth), put it into `inertial_poser/DynaIP/weights/weights.pt`

### Prepare test datasets

1. Download MagIMU dataset from [here](https://drive.google.com/file/d/1xh5JR40reX0zpeOW0XqVV2f7qtYRyzeX/view?usp=drive_link)
2. Put the unziped data into `datasets/MagIMU`. The structure should be like:

```
datasets
└── MagIMU
    ├── gt
    	├── xxx.pt
    ├── raw_measurements
    	├── xxx.pt
    ├── readings_noitom
    	├── xxx.pt
    └── align.py
```

### Run evaluation

The evaluation configurations are saved in `exp`. For each inertial poser, we provide three settings: `ESKF9`, `ESKF9+detector`, `ESKF9+detector+corrector`.

To get the mocap result, run:

```
python predict.py -c "exp/{INERTIAL_POSER_NAME}/{CONFIG}"
```

To get the evaluate results, run:

```
python eval.py
```

### Evaluation Results

With a successful run, the quantitative results should be as follows:

for `PNP`:

| method        | SIP Error (deg) | Angle Error (deg) | Joint Error (cm) | Vertex Error (cm) |
| ------------- | --------------- | ----------------- | ---------------- | ----------------- |
| eskf9         | 27.02           | 25.055            | 9.06             | 10.93             |
| eskf9+det     | 25.95           | 23.74             | 8.67             | 10.48             |
| eskf9+det+cor | **23.83**       | **19.89**         | **7.98**         | **9.50**          |

for `DynaIP`:

| method        | SIP Error (deg) | Angle Error (deg) | Joint Error (cm) | Vertex Error (cm) |
| ------------- | --------------- | ----------------- | ---------------- | ----------------- |
| eskf9         | 31.90           | 29.30             | 8.94             | 11.00             |
| eskf9+det     | 30.97           | 27.08             | 8.65             | 10.55             |
| eskf9+det+cor | **28.57**       | **21.39**         | **7.88**         | **9.56**          |

These results are not entirely consistent with those reported in the original paper. This is due to two reasons: 1) We fixed a minor bug in the timestamp down-sampling process; 2) We added an 2d embedding to the output layer of the corrector.
