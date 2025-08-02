# hash:sha256:e756a0b883f9adadcad36651ffd802e179366f9f54326ad48437b42c6adbbcca
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    aind-data-schema==2.0.0 \
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
    zarr==2.17.0 \
    -e git+https://github.com/AllenNeuralDynamics/abc-merfish-analysis@5b90fbd44590231ff657bec6de1db9ad7cb68375#egg=abc-merfish-analysis

COPY postInstall /
RUN /postInstall
