# hash:sha256:538e2bc7e15f103d2562a967c5dabda340dab7d4fdfa4a9b1ed102a63d8d6549
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    anndata==0.10.9 \
    colorcet==3.0.1 \
    colour-science==0.4.4 \
    nibabel==5.1.0 \
    openpyxl==3.1.5 \
    plotly==6.0.0 \
    pydantic==2.7.1 \
    scanpy==1.9.3 \
    scikit-learn==1.2.2 \
    scipy==1.9.1 \
    seaborn==0.13.2 \
    streamlit==1.38.0 \
    streamlit-aggrid==1.0.5 \
    zarr==2.17.0 \
    -e git+https://github.com/AllenNeuralDynamics/abc-merfish-analysis@d231e035d0052f3665481e49e9d55fe53c07c5b3#egg=abc-merfish-analysis

COPY postInstall /
RUN /postInstall
