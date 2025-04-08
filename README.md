# THALMANAC: Thalamus MERFISH analysis and access

Code Ocean capsule and streamlit app for exploration, analysis, and easy access of thalamus MERFISH data. Jupyter notebooks in `code` contain walkthroughs of the major functionality, while those in `code/figures` directly reproduce manuscript figures.


## Usage

As a streamlit app: start a Code Ocean cloud workstation in streamlit mode (or from the terminal on another cloud workstation, run `streamlit run /code/streamlit_app.py`)

As a package: install the core functionality as a package directly from github via pip: `pip install git+https://github.com/AllenNeuralDynamics/abc-merfish-analysis`

On Code Ocean: to work with your own copy of the capsule, select **Duplicate** from the **Capsule** menu and use the default option ("Link to git repository"). You can then make your own changes and sync back and forth from our shared github repository (https://github.com/AllenNeuralDynamics/thalmanac) via git. You can use git functions either in the **Reproducibility** panel on the right-hand side of the capsule view, or within the cloud workstation (more flexibility).

Locally: you can clone directly from github to your personal machine, but data assets will not be available. Code Ocean also has an option to download the capsule with data assets, but this will be very large if you're not careful!

### Pipeline

A few of the included analyses depend on preprocessed analysis results from separate capsules, due to complex dependencies that could not be included directly (specifically, NSF and SpaGCN methods). These preprocessed results are attached as data assets here, but for full end-to-end reproducibility a CO pipeline is available that links the execution of those preprocessing steps as inputs to a full re-run of the visualization/analysis notebooks: https://codeocean.allenneuraldynamics.org/capsule/9649481/tree.

# Code Ocean basics

## Code, Data, Results
Your capsule starts with the folders `/code`, `/data`, and `/results`. See our help article on [Paths](https://help.codeocean.com/getting-started/uploading-code-and-data/paths) for more information.

You can upload files to the `/code` or `/data` folders using the buttons at the top of the left-side **Files pane**.

Any plots, figures, and results data should be saved in the `/results` directory. At the end of each run, these files will appear in the right-side **Reproducibility pane**, where you can view and download them. When you publish a capsule, the most recent set of results will be preserved as part of the capsule.

## Environment and Dependencies

Click **Environment** on the left to find a computational environment to accommodate your software (languages, frameworks) or hardware (GPU) requirements. You can then further customize the environment by installing additional packages. The changes you make will be applied the next time your capsule runs. See our help articles on [the computational environment](https://help.codeocean.com/getting-started/the-computational-environment/configuring-your-computational-environment-an-overview) for more information.

### Environment Caching

The next time you run your capsule after making changes to any part of the environment, a custom environment will be built and cached for future runs.

When you publish a capsule, its computational environment will be preserved with it, thereby ensuring computational reproducibility.

## Troubleshooting

If you run into any issues or have any questions, we're here to help. You can reach out to us on on live chat, or, if you prefer, write to [support@codeocean.com](mailto:support@codeocean.com).

Use the **Help** menu above to explore the different sections of our [knowledge base](https://help.codeocean.com), with answers to many common questions.
