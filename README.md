# Multi-modal learning for geospatial vegetation forecasting
## _Code for Benson et. al., CVPR (2024)_

[![arXiv](https://img.shields.io/badge/arXiv-2303.16198-b31b1b.svg)](https://arxiv.org/abs/2303.16198) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10793870.svg)](https://doi.org/10.5281/zenodo.10793870)

:zap: Our paper on [_Multi-modal learning for geospatial vegetation forecasting_]([https://arxiv.org/abs/2303.16198v2](https://openaccess.thecvf.com/content/CVPR2024/html/Benson_Multi-modal_Learning_for_Geospatial_Vegetation_Forecasting_CVPR_2024_paper.html)) has been accepted to CVPR! We benchmark a wide range of EarthNet models on the new GreenEarthNet dataset, plus introducing a new transformer-based SOTA: Contextformer.

# Installation

The easiest is to just create a conda environment as follows:

```bash
conda create -n greenearthnet python=3.10
conda activate greenearthnet
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c conda-forge u8darts-notorch
conda install -c conda-forge numpy matplotlib pillow xarray zarr netcdf4 scipy imageio networkx scikit-image s3fs pyproj pyarrow seaborn cartopy tensorboard dask pytorch-lightning=1.7.7 torchmetrics=0.7.3 statsmodels
pip install earthnet earthnet-minicuber segmentation-models-pytorch albumentations
pip install git+https://github.com/earthnet2021/earthnet-models-pytorch.git@v0.1.0
```

# GreenEarthNet Dataset Download

We use the EarthNet Toolkit to download the GreenEarthNet dataset. First make sure you have it installed (`pip install earthnet`) and also make sure you have enough free disk space! We recommend 1TB. Then, downloading the dataset or a part thereof is as simple as:

```python
import earthnet as entk
entk.download(dataset = "greenearthnet", split = "train", save_directory = "data_dir")
```
Where  `data_dir` is the directory where EarthNet2021 shall be saved and `split` is `"all"`or a subset of `["train","iid","ood","extreme","seasonal"]`.

> **_NOTE:_**  In the paper the dataset is called **GreenEarthNet**, but in the codebase you will also find the following acronyms that were used during development: `earthnet2021x`, `en21x`.

# Model Inference

For training and inference, we are using `earthnet-models-pytorch`, which means this codebase only contains a small python script and then a lot of config files, alongside with the pre-trained weights stored on Zenodo (see below).

```bash
python test.py model_configs/path/to/config.yaml weights/path/to/weights.ckpt --track ood-t_chopped --pred_dir preds/path/to/save/predictions --data_dir data/path/to/dataset
```

You may also evaluate the model predictions and compute the metrics reported in the paper:
```bash
python eval.py path/to/test/dataset path/to/model/predictions path/for/saving/scores
```
You can optionally include a comparison directory, to also compute the outperformance score, e.g. relative to the Climatology baseline as done in the paper: `--compare_dir path/to/climatology/predictions`

# Model Training

```bash
python train.py model_configs/path/to/config.yaml --data_dir data/path/to/dataset
```

# Downloading Pre-Trained Model Weights

The pre-trained model weights are stored on [Zenodo](https://doi.org/10.5281/zenodo.10793870). You may download them (2.3GB) using this code:

```bash
wget https://zenodo.org/records/10793870/files/model_weights.zip
unzip model_weights.zip
```

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10793870.svg)](https://doi.org/10.5281/zenodo.10793870)


# Dataset Generation

The GreenEarthNet dataset has been generated with the EarthNet minicuber (`pip install earthnet-minicuber`). You can create more minicubes in the same style using the following code snippet:

```python
import earthnet_minicuber as emc

specs = {
    "lon_lat": (43.598946, 3.087414),  # center pixel
    "xy_shape": (128, 128),  # width, height of cutout around center pixel
    "resolution": 20,  # in meters.. will use this together with grid of primary provider..
    "time_interval": "2018-01-01/2021-12-31",
    "providers": [
        {
            "name": "s2",
            "kwargs": {
                "bands": ["B02", "B03", "B04", "B8A", "SCL"],  # , "B09", "B11", "B12"],
                "best_orbit_filter": False,
                "five_daily_filter": True,
                "brdf_correction": False,
                "cloud_mask": True,
                "cloud_mask_rescale_factor": 2,
                "aws_bucket": "planetary_computer",
            },
        },
        {"name": "srtm", "kwargs": {"bands": ["dem"]}},
        {"name": "alos", "kwargs": {}},
        {"name": "cop", "kwargs": {}},
        {
            "name": "esawc",
            "kwargs": {"bands": ["lc"], "aws_bucket": "planetary_computer"},
        },
        # {
        #     "name": "geom",
        #     "kwargs": {"filepath": "downloads/Geomorphons/geom/geom_90M_africa_europe.tif"}
        # }
        # Also Missing here: EOBS v26 https://surfobs.climate.copernicus.eu/dataaccess/access_eobs.php#datafiles
    ],
}

mc = emc.load_minicube(specs, compute = True)
```

To fully obtain minicubes aligned with the GreenEarthNet dataset that are useful for prediction, you still need to download and merge [E-OBS v26](https://surfobs.climate.copernicus.eu/dataaccess/access_eobs.php#datafiles).

# Cloud Mask Training

The Cloud Mask Algorithm is trained on the [CloudSEN12](https://cloudsen12.github.io/) dataset, if you want to repeat the training, follow the instructions on their website to download the dataset and then run the python script:
```python
python cloudmask/train_cloudmask_l2argbnir.py
```
> **_NOTE:_**  This python script is based upon the original [CloudSEN12 MobileNetv2 implementation](https://github.com/cloudsen12/models/blob/master/unet_mobilenetv2/cloudsen12_unet.py)!

If you wish to just use the trained Cloud Mask Algorithm, you may do so, it is included in the [EarthNet Minicuber](https://github.com/earthnet2021/earthnet-minicuber) package (`pip install earthnet-minicuber`).

# Reproducing Paper Tables

## Table 2
| Model | Config/Weights |
| --- | --- |
| Persistence | `python model_pixelwise/persistence.py --data_dir data/path/to/dataset --out_dir experiments/path/to/save/predictions` |
| Previous Year | `python model_pixelwise/previousyear.py --data_dir data/path/to/dataset --out_dir experiments/path/to/save/predictions` |
| Climatology | `python model_pixelwise/climatology.py --data_dir data/path/to/dataset --out_dir experiments/path/to/save/predictions` |
| Kalman Filter | `python model_pixelwise/predict_kalman_xgb_prophet.py minicube_idx --data_dir data/path/to/dataset --out_dir experiments/path/to/save/predictions` |
| LightGBM | `python model_pixelwise/predict_kalman_xgb_prophet.py minicube_idx --data_dir data/path/to/dataset --out_dir experiments/path/to/save/predictions` |
| Prophet | `python model_pixelwise/predict_kalman_xgb_prophet.py minicube_idx --data_dir data/path/to/dataset --out_dir experiments/path/to/save/predictions` |
| Diaconu ConvLSTM | [See Weather2Land Github](https://github.com/dcodrut/weather2land) |
| Kladny SG-ConvLSTM | [See SatelliteImageForecasting Github](https://github.com/rudolfwilliam/satellite_image_forecasting) |
| Earthformer EarthNet2021 weights | [See Earthformer Github](https://github.com/amazon-science/earth-forecasting-transformer/tree/main/scripts/cuboid_transformer/earthnet_w_meso) |
| ConvLSTM Seed <42, 97, 27> | `model_configs/convlstm/convlstm1M/seed=<42,97,27>.<yaml/ckpt>` |
| Earthformer retrained | [See Earthformer Github](https://github.com/gaozhihan/earth-forecasting-transformer/tree/earthnet2021x/scripts/cuboid_transformer/earthnet2021x/cond_weather_data) |
| PredRNN Seed <42, 97, 27> | `model_configs/predrnn/predrnn1M/seed=<42,97,27>.<yaml/ckpt>` |
| SimVP Seed <42, 97, 27> | `model_configs/simvp/simvp6M/seed=<42,97,27>.<yaml/ckpt>` |
| Contextformer Seed <42, 97, 27> | `model_configs/contextformer/contextformer6M/seed=<42,97,27>.<yaml/ckpt>` |

## Table 3
| Model | Config/Weights |
| --- | --- |
| Climatology | |
| 1x1 LSTM | `model_configs/lstm1x1/<lstm1x1,spatialshuffle>/seed=42.<yaml/ckpt>` |
| Next-frame UNet | `model_configs/nfunet/<nfunet,spatialshuffle>/seed=42.<yaml/ckpt>` |
| Next-cuboid UNet | `model_configs/nfunet/<nfunet,spatialshuffle>/seed=42.<yaml/ckpt>` |
| ConvLSTM | `model_configs/convlstm/<convlstm1M,spatialshuffle>/seed=42.<yaml/ckpt>` |
| PredRNN | `model_configs/predrnn/<predrnn1M,spatialshuffle>/seed=42.<yaml/ckpt>` |
| SimVP | `model_configs/simvp/<simvp6M,spatialshuffle>/seed=42.<yaml/ckpt>` |
| Contextformer | `model_configs/contextformer/<contextformer6M,spatialshuffle>/seed=42.<yaml/ckpt>` |

## Table 4
| Model | Config/Weights |
| --- | --- |
| MLP vision encoder | `model_configs/contextformer/mlp_vision_enc/seed=42.<yaml/ckpt>` |
| PVT encoder (frozen) | `model_configs/contextformer/pvt_vision_frozen/seed=42.<yaml/ckpt>` |
| w/ cloud mask token | `model_configs/contextformer/mask_clouds/seed=42.<yaml/ckpt>` |
| w/ learned Vhat0 | `model_configs/contextformer/learned_vhat0/seed=42.<yaml/ckpt>` |
| w/ last pixel Vhat0 | `model_configs/contextformer/last_vhat0/seed=42.<yaml/ckpt>` |
| Contextformer 6M | `model_configs/contextformer/contextformer6M/seed=<42,97,27>.<yaml/ckpt>` |
| Contextformer 16M | `model_configs/contextformer/contextformer16M/seed=<42,97,27>.<yaml/ckpt>` |

## Table 5

| Model | Config/Weights |
| --- | --- |
| Climatology | |
| Contextformer 6M | `model_configs/contextformer/contextformer6M/seed=<42,97,27>.<yaml/ckpt>` |

# Reproducing Paper Figures

## Figure 3 and 5

| Model | Config/Weights |
| --- | --- |
| Contextformer 6M | `model_configs/contextformer/contextformer6M/seed=42.<yaml/ckpt>` |

## Figure 4

| Model | Config/Weights |
| --- | --- |
| ConvLSTM | `model_configs/convlstm/<convlstm1M,noweather>/seed=<42,97,27>.<yaml/ckpt>` |
| PredRNN | `model_configs/predrnn/<predrnn1M,noweather>/seed=<42,97,27>.<yaml/ckpt>` |
| SimVP | `model_configs/simvp/<simvp6M,noweather>/seed=<42,97,27>.<yaml/ckpt>` |
| Contextformer | `model_configs/contextformer/<contextformer6M,noweather>/seed=<42,97,27>.<yaml/ckpt>` |

# Citation

```bibtex
@InProceedings{Benson_2024_CVPR,
    author    = {Benson, Vitus and Robin, Claire and Requena-Mesa, Christian and Alonso, Lazaro and Carvalhais, Nuno and Cort\'es, Jos\'e and Gao, Zhihan and Linscheid, Nora and Weynants, M\'elanie and Reichstein, Markus},
    title     = {Multi-modal Learning for Geospatial Vegetation Forecasting},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {27788-27799}
}
```
