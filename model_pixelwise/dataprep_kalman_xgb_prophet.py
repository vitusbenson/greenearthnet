import warnings
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

warnings.filterwarnings("ignore")


def dataframe_from_minicube_pixel(mc, lat_idx, lon_idx):

    pixel = mc.isel(lat=lat_idx, lon=lon_idx)
    nir = pixel.s2_B8A
    red = pixel.s2_B04

    ndvi = (nir - red) / (nir + red + 1e-8)

    pixel["s2_ndvi"] = ndvi

    sen2_mask = pixel.s2_dlmask.where(
        pixel.s2_dlmask > 0, 4 * (~pixel.s2_SCL.isin([1, 2, 4, 5, 6, 7]))
    )
    sen2_data = (
        pixel[[f"s2_{b}" for b in ["ndvi", "B02", "B03", "B04", "B8A"]]]
        .where(sen2_mask == 0, np.NaN)
        .isel(time=slice(4, None, 5))
    )

    eobs = pixel[
        [f"eobs_{v}" for v in ["hu", "pp", "qq", "rr", "tg", "tn", "tx"]]
    ]  #'fg',

    eobs_data = {}
    for var in [
        f"eobs_{v}" for v in ["hu", "pp", "qq", "rr", "tg", "tn", "tx"]
    ]:  #'fg',
        eobs_data[f"{var}_mean"] = eobs[var].coarsen(time=5, coord_func="max").mean()
        eobs_data[f"{var}_min"] = eobs[var].coarsen(time=5, coord_func="max").min()
        eobs_data[f"{var}_max"] = eobs[var].coarsen(time=5, coord_func="max").max()
        eobs_data[f"{var}_std"] = eobs[var].coarsen(time=5, coord_func="max").std()
    eobs_5d = xr.Dataset(eobs_data)

    all_data = xr.merge(
        [
            sen2_data,
            eobs_5d,
            pixel[["nasa_dem", "alos_dem", "cop_dem", "esawc_lc", "geom_cls"]],
        ]
    )

    return all_data.to_dataframe().drop(columns=["sentinel:product_id"])


# Extra single dataframes
# Then concat them all into a big dataframe
# Then darts from_group_dataframe with group_cols=[lat, lon]
# Then Should have data ready.... :))

# ~100000 groups = 5GB ??


def process_one_minicube(mc_path):

    mc = xr.open_dataset(mc_path)

    dfs = []
    for lc in [10.0, 20.0, 30.0, 40.0]:

        lat_idxs, lon_idxs = np.where(mc.esawc_lc == lc)

        if len(lat_idxs) > 0:
            idx = np.random.randint(len(lat_idxs))

            df = dataframe_from_minicube_pixel(mc, lat_idxs[idx], lon_idxs[idx])

            df["id"] = f"{mc_path.stem}_{int(lc)}_{idx}"

            dfs.append(df)

    return pd.concat(dfs).reset_index()


def create_train_dataframe(data_dir, save_dir):

    data_dir = Path(data_dir)

    mc_paths = sorted(list(data_dir.glob("**/*.nc")))  # [:20]

    with ProcessPoolExecutor(max_workers=20) as pool:
        dfs = list(tqdm(pool.map(process_one_minicube, mc_paths), total=len(mc_paths)))

    df = pd.concat(dfs).reset_index()

    save_dir = Path(save_dir)

    df.to_parquet(save_dir / "en21x_darts_train.parquet.gzip")


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/greenearthnet/",
        metavar="path/to/dataset",
        help="Path where dataset is located",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/greenearthnet/",
        metavar="path/to/dataset",
        help="Path where dataset will be saved",
    )
    args = parser.parse_args()

    create_train_dataframe(args.data_dir, args.out_dir)
