# hash:sha256:c6a54b1eb5c982b04110c51833b4f94e0bd1d765437840ba77e4a791b237b450
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
    -e git+https://github.com/AllenNeuralDynamics/abc-merfish-analysis@74ae0a5d67c014fe931f8e5f89e8179f2c57aba0#egg=abc-merfish-analysis

COPY postInstall /
RUN /postInstall
