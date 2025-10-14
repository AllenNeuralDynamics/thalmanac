# THALMANAC: Thalamus MERFISH analysis and access

Code Ocean capsule and streamlit app for exploration, analysis, and easy access of thalamus MERFISH data from the Allen Brain Cell Atlas. Jupyter notebooks in `code` contain walkthroughs of the major functionality, while those in `code/figures` directly reproduce manuscript figures. Scripts in `code/scripts` include those used during the analysis to reformat label volumes and nucleus annotations, and one to save processed data in Cirrocumulus-compatible h5ad format (`scripts/abc_th_to_cirro.py`). See manuscript methods for more details. 

## Input data
- _Allen Brain Cell (ABC) Atlas_: This capsule uses the thalamus subset of the mouse whole-brain transcriptomic cell type atlas (Hongkui Zeng) dataset,
loaded following https://alleninstitute.github.io/abc_atlas_access/intro.html (either via the Code Ocean asset that mounts s3 data directly, or by downloading to a local cache).

- _Domain detection outputs_: _merfish_638850_nsf_ and _merfish_638850_spagcn_

- _Common Coordinate Framework (CCF) images_: Alternate brain region label volumes, including _KimLabDevCCFv001_, _KimLabDevCCFv001_ (resampled to match ABC atlas sections), _CCF-templates-realigned_ (volumes from CCFv3 and DevCCFv001 matched to ABC atlas sections via the alternate manual affine alignment stored in _thalamus_realignment_). These are included for visualization functionality only, not used to generate manuscript figures.

## Usage

As a streamlit app: go to https://thalmanac.allenneuraldynamics.org/ or start a Code Ocean cloud workstation in streamlit mode.

As a package: install the core functionality as a package directly from github via pip: `pip install git+https://github.com/AllenNeuralDynamics/abc-merfish-analysis`

On Code Ocean:

To reproduce results exactly, use the reproducible run functionality - in the App Builder pane, you can specify the path to any notebook or script, then select Run to execute it and view results.

To edit your own copy, select **Duplicate** from the **Capsule** menu and use the default option ("Link to git repository"). You can then make your own changes and sync back and forth from our shared github repository (https://github.com/AllenNeuralDynamics/thalmanac) via git. You can use git functions either in the **Reproducibility** panel on the right-hand side of the capsule view, or within the cloud workstation (more flexibility).

Locally: select **Export** from the **Capsule** menu to download all code together with instructions for running locally via docker.

### Pipeline

A few of the included analyses depend on preprocessed analysis results from separate capsules, due to complex dependencies that could not be included directly (specifically, NSF and SpaGCN methods). These preprocessed results are attached as data assets here, but for full end-to-end reproducibility a CO pipeline is available that links the execution of those preprocessing steps as inputs to a full re-run of the visualization/analysis notebooks: https://codeocean.allenneuraldynamics.org/capsule/9649481/tree.

