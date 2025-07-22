# hash:sha256:928932b1cc88c3e9565f795152d603031347fb5993e4de2290c8edf38ca0380e
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

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
    pydantic==2.7.1 \
    scanpy==1.9.3 \
    scikit-learn==1.2.2 \
    scipy==1.9.1 \
    seaborn==0.13.2 \
    streamlit==1.46.0 \
    streamlit-aggrid==1.0.5 \
    zarr==2.17.0

COPY postInstall /
RUN /postInstall
