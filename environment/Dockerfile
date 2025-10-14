# hash:sha256:562286627a20a65407a0762c44a41e6c1f0ad94ad075be906450a340e758e2fe
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip install -U --no-cache-dir \
    aind-data-schema==2.0.0 \
    anndata==0.10.9 \
    colorcet==3.0.1 \
    colour-science==0.4.4 \
    matplotlib==3.10.5 \
    networkx==3.4.2 \
    nibabel==5.1.0 \
    numpy==1.24.4 \
    openpyxl==3.1.5 \
    pandas==2.3.1 \
    pyarrow==21.0.0 \
    pydantic==2.7.1 \
    scanpy==1.9.3 \
    scikit-learn==1.2.2 \
    scipy==1.9.1 \
    seaborn==0.13.2 \
    streamlit==1.46.0 \
    streamlit-aggrid==1.0.5 \
    zarr==2.17.0 \
    -e git+https://github.com/AllenNeuralDynamics/abc-merfish-analysis@fdd6ba5e66505f7e9e639cf15c8a6654ac01ccf3#egg=abc-merfish-analysis

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
