import subprocess
from datetime import datetime

subprocess.run(["pip", "install", "spatialdata==0.2.3", "dask==2024.4.1", "xarray==2024.07.0"])

import spatialdata as sd
import pandas as pd
import numpy as np
import nibabel
import abc_merfish_analysis
from abc_merfish_analysis import abc_load as abc
from abc_merfish_analysis import ccf_registration as ccf
from abc_merfish_analysis import ccf_transforms as ccft
from abc_merfish_analysis import ccf_images as ccfi
from pathlib import Path

realignment_files = Path("/data/thalamus_realignment")
df_full = abc.get_combined_metadata(drop_unused=False)
# permissive spatial subset using published alignment
# (previously using manual subset)
df = abc.filter_by_thalamus_coords(df_full, buffer=25, include_white_matter=True)

coords = ["x_section", "y_section", "z_section"]
slice_label = "slice_int"
df[slice_label] = df["z_section"].apply(lambda x: int(x * 10))

transforms_by_section = ccf.read_quicknii_file(
    realignment_files / "quicknii_refined_20240228.json",
    scale=25,
)
minmax = pd.read_csv(
    realignment_files/ "brain3_thalamus_coordinate_bounds.csv",
    index_col=0,
)

# load to spatialdata
norm_transform = ccft.get_normalizing_transform(
    min_xy=minmax.loc["min"].values, max_xy=minmax.loc["max"].values, flip_y=True
)
cells_by_section = ccft.parse_cells_by_section(
    df, transforms_by_section, norm_transform, coords, slice_label=slice_label
)
sdata = sd.SpatialData.from_elements_dict(cells_by_section)
# need to write for some functionality to work?
sdata.write("/scratch/abc_atlas_realigned.zarr", overwrite=True)

# transform
transformed_points = pd.concat(
    (df.compute() for df in sdata.transform_to_coordinate_system("ccf").points.values())
)

# update dataframe
new_coords = [f"{x}_ccf_realigned" for x in "xyz"]  # xyz order
df = df.join(
    transformed_points[list("xyz")].rename(columns=dict(zip("xyz", new_coords)))
)


ngrid = 1100
nz = 76
z_res = 2


def transform_section(section, imdata=None, fname=None):
    target = sdata[section]
    source = sdata[fname]
    scale = 10e-3
    target_img, target_grid_transform = ccft.map_image_to_slice(
        sdata, imdata, source, target, scale=scale, ngrid=ngrid, centered=False
    )
    return target_img


# from multiprocessing import Pool
# import functools
def save_resampled_image(imdata, fname):
    dtype = np.int64
    img_transform = sd.transformations.Scale(10e-3 * np.ones(3), "xyz")
    labels = sd.models.Labels3DModel.parse(
        imdata, dims="xyz", transformations={"ccf": img_transform}
    )
    sdata.labels[fname] = labels
    img_stack = np.zeros((ngrid, ngrid, nz), dtype=dtype)

    for section in sdata.points.keys():
        target_img = transform_section(section, imdata=imdata, fname=fname)
        i = int(np.rint(int(section) / z_res))
        img_stack[:, :, i] = target_img.T
    # with Pool(processes=8) as p:
    #     out = p.map(functools.partial(transform_section, imdata=imdata, fname=fname),
    #                 sdata.points.keys())
    # out = map(functools.partial(transform_section, imdata=imdata, fname=fname),
    #                 sdata.points.keys())
    # img_stack = np.stack(out, axis=-1)

    nifti_img = nibabel.Nifti1Image(img_stack, affine=np.eye(4), dtype=dtype)
    nibabel.save(nifti_img, f"/results/{fname}.nii.gz")


# CCFv3
imdata = abc.get_ccf_labels_image(resampled=False)
df["parcellation_index_realigned"] = imdata[
    ccfi.image_index_from_coords(df[new_coords])
]
save_resampled_image(imdata, "abc_realigned_ccf_labels")

# add parcellation metadata
ccf_df = abc._ccf_metadata
ccf_df = ccf_df.pivot(
    index="parcellation_index",
    columns="parcellation_term_set_name",
    values="parcellation_term_acronym",
).astype("category")
df = df.join(
    ccf_df[["division", "structure", "substructure"]].rename(
        columns=lambda x: f"parcellation_{x}_realigned"
    ),
    on="parcellation_index_realigned",
)

# Kim Lab DevCCF
imdata = abc.get_ccf_labels_image(resampled=False, devccf=True)
df["parcellation_index_realigned_devccf"] = imdata[
    ccfi.image_index_from_coords(df[new_coords])
]
save_resampled_image(imdata, "abc_realigned_devccf_labels")

devccf_index = abc._get_devccf_metadata()
df["parcellation_devccf"] = df["parcellation_index_realigned_devccf"].map(
    devccf_index.to_dict()
)

df.to_parquet("/results/abc_realigned_metadata_thalamus-boundingbox.parquet")

import aind_data_schema.core.processing as ps
from aind_data_schema.components.identifiers import DataAsset
import aind_data_schema.core.data_description as ds
from aind_data_schema.core.metadata import Metadata

dt = datetime.now()
name = f"CCF-templates-resampled_{dt.strftime('%Y-%m-%d_%H-%M-%S')}"
md = Metadata(
    name=name,
    location=f"s3://aind-open-data/{name}",
    data_description=ds.DataDescription(
        name=name,
        creation_time=dt,
        institution=ds.Organization.AIND,
        funding_source=[ds.Funding(
            funder=ds.Organization.NINDS,
            grant_number="U19NS123714",
            fundee=[ds.Person(name="Karel Svoboda")]
        )],
        data_level=ds.DataLevel.DERIVED,
        investigators=[ds.Person(name="Thomas Chartrand")],
        project_name="Thalamus in the middle - Project 2 Cell-type specific thalamocortical projectome",
        # add other CCF atlas modalities?
        modalities=[ds.Modality.MERFISH],
    ),
    processing=ps.Processing.create_with_sequential_process_graph(
        data_processes=[
            ps.DataProcess(
                process_type=ps.ProcessName.OTHER,
                name="Create rasterized images",
                stage=ps.ProcessStage.ANALYSIS,
                experimenters=["Thomas Chartrand"],
                start_date_time=datetime(2023,2,28),
                end_date_time=datetime(2023,2,28),
                code=ps.Code(
                    name="THALMANAC tools",
                    url="https://github.com/AllenNeuralDynamics/thalmanac",
                    run_script="code/scripts/export_quicknii_images.py",
                    input_data=[DataAsset(url="s3://allen-brain-cell-atlas")]
                ),
                notes="""Create thalamus-focused rasterized images of MERFISH sections, colored
                    by subclass/supertype, for alignment to CCF reference"""
            ),
            ps.DataProcess(
                process_type=ps.ProcessName.IMAGE_ATLAS_ALIGNMENT,
                name="Manual CCF alignment",
                stage=ps.ProcessStage.ANALYSIS,
                experimenters=["Thomas Chartrand"],
                start_date_time=datetime(2024,2,28),
                end_date_time=datetime(2024,2,28),
                code=ps.Code(
                    name="QuickNII",
                    url="https://github.com/Neural-Systems-at-UIO/QuickNII"
                ),
                output_path="quicknii_refined_20240228.json"
            ),
            ps.DataProcess(
                process_type=ps.ProcessName.OTHER,
                name="Reverse transform atlas",
                stage=ps.ProcessStage.ANALYSIS,
                experimenters=["Thomas Chartrand"],
                start_date_time=dt,
                end_date_time=dt,
                code=ps.Code(
                    name="THALMANAC tools",
                    url="https://github.com/AllenNeuralDynamics/thalmanac",
                    run_script="code/scripts/ingest_quicknii_transforms.py",
                    input_data=[DataAsset(url="s3://allen-brain-cell-atlas")]
                ),
                notes="""Create resampled section images from input atlas label volumes, one for each",
                    "MERFISH section of interest"""
            ),
        ]
    )
)
md.write_standard_file("/results")
md.processing.write_standard_file("/results")
md.data_description.write_standard_file("/results")