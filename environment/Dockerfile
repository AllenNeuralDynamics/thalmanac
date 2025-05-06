# hash:sha256:6090ba66e5ec0ee286a6310f50ba66649b747633f64f6b4941ba537a874f1682
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
    zarr==2.17.0

COPY postInstall /
RUN /postInstall
