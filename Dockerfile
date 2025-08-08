# Use Python 3.12 slim image
FROM python:3.12-slim

# Install required system packages
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    wget \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install huggingface CLI
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir huggingface_hub

# Set working directory
WORKDIR /app

# Clone OmniParser repo
RUN git clone https://github.com/microsoft/OmniParser.git .

# Install Python dependencies and downgrade transformers to fix generate method
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir transformers==4.51.3

# Download weights (icon_detect and icon_caption) - OmniParser's fine-tuned models
RUN rm -rf weights/icon_detect weights/icon_caption weights/icon_caption_florence && \
    mkdir -p weights && \
    for folder in icon_caption icon_detect; do \
        huggingface-cli download microsoft/OmniParser-v2.0 \
            --local-dir weights \
            --repo-type model \
            --include "$folder/*"; \
    done && \
    mv weights/icon_caption weights/icon_caption_florence

# Set working directory to the server
WORKDIR /app/omnitool/omniparserserver

# Expose default port
EXPOSE 8000

# Start the OmniParser server
CMD ["python", "-m", "omniparserserver", "--device", "gpu", "--BOX_TRESHOLD", "0.02"]
