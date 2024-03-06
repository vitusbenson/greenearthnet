import logging
import warnings
from pathlib import Path

import darts
import numpy as np
import pandas as pd
import xarray as xr
from darts import TimeSeries
from darts.models import (
    ARIMA,
    KalmanForecaster,
    LightGBMModel,
    Prophet,
    RegressionModel,
    StatsForecastAutoARIMA,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")
logger = logging.getLogger("cmdstanpy")
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)


def dataframe_from_minicube(mc):

    first_sen2_idx = np.where(mc.s2_avail == 1)[0][0]
    last_sen2_idx = np.where(mc.s2_avail == 1)[0][-1]

    pixel = mc.isel(
        time=slice(first_sen2_idx + 1, last_sen2_idx + 1)
    )  # .isel(lat = lat_idx, lon = lon_idx)
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

    return all_data  # .to_dataframe().drop(columns = ["sentinel:product_id"])


def predict_one_pixel(df, targ_time, use_lgbm=True, use_kalman=True, use_prophet=False):

    ndvi = TimeSeries.from_dataframe(
        df.reset_index(), time_col="time", value_cols=["s2_ndvi"]
    )
    future_x = TimeSeries.from_dataframe(
        df.reset_index(),
        time_col="time",
        value_cols=[
            "eobs_hu_mean",
            "eobs_hu_min",
            "eobs_hu_max",
            "eobs_hu_std",
            "eobs_pp_mean",
            "eobs_pp_min",
            "eobs_pp_max",
            "eobs_pp_std",
            "eobs_qq_mean",
            "eobs_qq_min",
            "eobs_qq_max",
            "eobs_qq_std",
            "eobs_rr_mean",
            "eobs_rr_min",
            "eobs_rr_max",
            "eobs_rr_std",
            "eobs_tg_mean",
            "eobs_tg_min",
            "eobs_tg_max",
            "eobs_tg_std",
            "eobs_tn_mean",
            "eobs_tn_min",
            "eobs_tn_max",
            "eobs_tn_std",
            "eobs_tx_mean",
            "eobs_tx_min",
            "eobs_tx_max",
            "eobs_tx_std",
        ],
    )
    future_x = darts.utils.missing_values.fill_missing_values(
        future_x, interpolate_kwargs={"fill_value": "extrapolate"}
    )
    future_x = darts.utils.missing_values.fill_missing_values(future_x, fill=0.0)

    context, target = ndvi.split_before(targ_time)

    context = darts.utils.missing_values.fill_missing_values(
        context, interpolate_kwargs={"fill_value": "extrapolate"}
    )

    out = {}

    if use_lgbm:
        model = LightGBMModel(
            lags=10, lags_future_covariates=[-10, 0], output_chunk_length=20
        )
        model.fit(context, future_covariates=future_x)
        pred = model.predict(n=20, future_covariates=future_x)

        out["lgbm"] = (
            pred.data_array().to_dataset("component").squeeze().s2_ndvi.values
        )  # .assign_coords({"lon": df.iloc[0].lon, "lat": df.iloc[0].lat}).expand_dims(dim = ["lat", "lon"])

    if use_kalman:
        model = KalmanForecaster()
        model.fit(context, future_covariates=future_x)
        pred = model.predict(n=20, future_covariates=future_x)

        out["kalman"] = (
            pred.data_array().to_dataset("component").squeeze().s2_ndvi.values
        )  # .assign_coords({"lon": df.iloc[0].lon, "lat": df.iloc[0].lat}).expand_dims(dim = ["lat", "lon"])
    if use_prophet:
        model = Prophet()
        model.fit(context, future_covariates=future_x)
        pred = model.predict(n=20, future_covariates=future_x)

        out["prophet"] = (
            pred.data_array().to_dataset("component").squeeze().s2_ndvi.values
        )  # .assign_coords({"lon": df.iloc[0].lon, "lat": df.iloc[0].lat}).expand_dims(dim = ["lat", "lon"])

    return out


def predict_minicube(data_dir, save_dir, mc_name, models=["lgbm", "prophet", "kalman"]):

    data_dir = Path(data_dir)
    save_dir = Path(save_dir)

    mc_path = data_dir / "iidx" / mc_name

    mc = xr.open_dataset(mc_path)

    mc_processed = dataframe_from_minicube(mc)

    regions = sorted(
        [d.stem for d in (data_dir / "ood-t_chopped").iterdir() if d.is_dir()]
    )

    for region in tqdm(regions, desc="Region", position=0):
        targ_path = data_dir / "ood-t_chopped" / region / mc_name
        targ = xr.open_dataset(targ_path)
        targ_time = pd.Timestamp(targ.time.isel(time=50).dt.date.item())

        out_time = targ.time.isel(time=slice(54, None, 5)).time.drop_vars(
            ["latitude_eobs", "longitude_eobs", "sentinel:product_id"]
        )
        context_time = targ.time.isel(time=slice(4, 50, 5)).time.drop_vars(
            ["latitude_eobs", "longitude_eobs", "sentinel:product_id"]
        )

        H = len(mc.lat)
        W = len(mc.lon)

        all_preds = {}
        for model in models:
            all_preds[model] = np.full((H, W, len(out_time)), fill_value=np.NaN)

        for lat_idx in tqdm(range(H), desc="Lat", position=1, leave=False):
            for lon_idx in tqdm(range(W), desc="Lon", position=2, leave=False):

                natural_veg = (
                    mc_processed.esawc_lc.isel(lat=lat_idx, lon=lon_idx).item() < 41
                )
                curr_ndvi = mc_processed.s2_ndvi.isel(lat=lat_idx, lon=lon_idx).sel(
                    time=out_time
                )
                no_seas_water = curr_ndvi.min().item() > 0.0
                n_obs = (~np.isnan(curr_ndvi)).sum().item() >= 10
                sigma_targ = curr_ndvi.std("time").item() > 0.1
                context_ndvi = mc_processed.s2_ndvi.isel(lat=lat_idx, lon=lon_idx).sel(
                    time=context_time
                )
                n_obs_context = (~np.isnan(context_ndvi)).sum().item() >= 3

                if (
                    natural_veg
                    and no_seas_water
                    and n_obs
                    and n_obs_context
                    and sigma_targ
                ):

                    curr_df = (
                        mc_processed.isel(lat=lat_idx, lon=lon_idx)
                        .to_dataframe()
                        .drop(columns=["sentinel:product_id"])
                    )
                    # lat = mc.lat.isel(lat=lat_idx).item()
                    # lon = mc.lon.isel(lon=lon_idx).item()
                    out = predict_one_pixel(
                        curr_df,
                        targ_time,
                        use_lgbm=("lgbm" in models),
                        use_prophet=("prophet" in models),
                        use_kalman=("kalman" in models),
                    )

                    for model in out:
                        all_preds[model][lat_idx, lon_idx, :] = out[model]

        for model in all_preds:

            pred = xr.Dataset(
                {
                    "ndvi_pred": xr.DataArray(
                        data=all_preds[model],
                        coords={"time": out_time, "lat": mc.lat, "lon": mc.lon},
                        dims=["lat", "lon", "time"],
                    )
                }
            )

            out_path = (
                save_dir
                / f"local_{model}"
                / "preds"
                / "ood-t_chopped"
                / region
                / mc_name
            )
            out_path.parent.mkdir(exist_ok=True, parents=True)
            print(f"Saving to {out_path}")
            pred.to_netcdf(path=out_path, encoding={"ndvi_pred": {"dtype": "float32"}})

    return


def main(args):
    idx = args.idx

    data_dir = Path(args.data_dir)
    pred_dir = Path(args.out_dir)

    mc_names = sorted([p.name for p in (data_dir / "iidx").glob("*.nc")])

    mc_name = mc_names[idx]
    print(f"Predicting {mc_name}")
    predict_minicube(
        data_dir, pred_dir, mc_name, models=["lgbm", "kalman", "prophet"]
    )  # ["kalman"])#


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("idx", type=int, help="index of the minicube")
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
        default="experiments/greenearthnet/",
        metavar="path/to/experiments",
        help="Path where predictions will be saved",
    )
    args = parser.parse_args()
    main(args)
