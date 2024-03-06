# Multi-modal learning for geospatial vegetation forecasting
## _Code for Benson et. al., CVPR (2024)_


# TODO:
- [x] Add SimVP
- [ ] Merge Request EMP + version release
- [x] Upload Model weights somewhere (e.g. Zenodo?) + Make download script that puts them in the folder `model_weights`
- [x] Add timeseries Model code
- [ ] Put Markdown Table with Model + Weight for each Table / Figure in main text
- [ ] Add Dataset Generation Script
- [ ] Add Cloud Mask Training Script
- [ ] Add Model Eval Script
- [ ] Add Script to aggregate Data for each Figure/Table
- [ ] Add Installation guide
- [ ] Add Citation
- [ ] Update EarthNet website/documentation


# Installation

# GreenEarthNet Dataset Download

Ensure you have enough free disk space! We recommend 1TB.

```
import earthnet as entk
entk.download(dataset = "earthnet2021x", split = "train", save_directory = "data_dir")
```
Where  `data_dir` is the directory where EarthNet2021 shall be saved and `split` is `"all"`or a subset of `["train","iid","ood","extreme","seasonal"]`.

> **_NOTE:_**  In the paper the dataset is called **GreenEarthNet**, but in the codebase you will also find the following acronyms that were used during development: `earthnet2021x`, `en21x`.

# Model Inference

For training and inference, we are using `earthnet-models-pytorch`, which means this codebase only contains a small python script and then a lot of config files, alongside with the pre-trained weights stored on Zenodo (see below).

```bash
python test.py model_configs/path/to/config.yaml weights/path/to/weights.ckpt --track ood-t_chopped --pred_dir preds/path/to/save/predictions --data_dir data/path/to/dataset
```

# Model Training

```bash
python test.py model_configs/path/to/config.yaml --data_dir data/path/to/dataset
```

# Downloading Pre-Trained Model Weights

The pre-trained model weights are stored on [Zenodo](https://doi.org/10.5281/zenodo.10790832). You may download them (2.2GB) using this code:

```bash
wget https://zenodo.org/records/10790832/files/model_weights.zip
unzip model_weights.zip
```

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10790832.svg)](https://doi.org/10.5281/zenodo.10790832)


# Cloud Mask Training

# Dataset Generation

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
| 1x1 LSTM | `model_configs/lstm1x1/<lstm1x1,spatialshuffle>/seed=42.yaml` |
| Next-frame UNet | `model_configs/nfunet/<nfunet,spatialshuffle>/seed=42.yaml` |
| Next-cuboid UNet | `model_configs/nfunet/<nfunet,spatialshuffle>/seed=42.yaml` |
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