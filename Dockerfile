FROM ghcr.io/triton-inference-server/server:24.03-py3

ENV MODEL_REPOSITORY=/models
RUN mkdir -p ${MODEL_REPOSITORY}
COPY triton_models/ ${MODEL_REPOSITORY}
EXPOSE 8000 8001 8002
CMD ["tritonserver", "--model-repository=/models"]
