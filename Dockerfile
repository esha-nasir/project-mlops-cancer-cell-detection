# Use Triton Inference Server base image
FROM eshanasir/tritonserver:24.03-py3

# Set environment variable for the model repository
ENV MODEL_REPOSITORY=/models

# Create model repository directory
RUN mkdir -p ${MODEL_REPOSITORY}

# Copy local models into the container - fix the path here
COPY brain_tumor_segmentation/triton_models/ ${MODEL_REPOSITORY}

# Expose Triton's ports
EXPOSE 8000 8001 8002

# Start the Triton server with your model repository
CMD ["tritonserver", "--model-repository=/models"]
