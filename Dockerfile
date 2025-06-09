# Slim Triton server base
FROM nvcr.io/nvidia/tritonserver:24.03-py3-min

ENV MODEL_REPOSITORY=/models

RUN mkdir -p ${MODEL_REPOSITORY}
VOLUME /models  # Use volume instead of COPY

EXPOSE 8000 8001 8002

CMD ["tritonserver", "--model-repository=/models"]
