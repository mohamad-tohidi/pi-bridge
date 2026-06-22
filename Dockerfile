# Use Python as the base image
FROM docker.arvancloud.ir/python:3.11-slim

# Install Node.js and npm
# We use the NodeSource setup script to get a recent version of Node.js
RUN apt-get update && apt-get install -y \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the entire repository into the container
# The .dockerignore file will ensure we don't copy node_modules or __pycache__
COPY . .

# 1. Install Node.js dependencies for the bridge component
# The bridge requires its own node_modules to be present
RUN cd bridge && npm install

# 2. Install the Python package and its dependencies
# This installs 'pi-bridge' and everything in pyproject.toml
RUN pip install --no-cache-dir .

# 3. Install the Pi Agent globally
# The PiSession class in pi_bridge/session.py looks for the agent in the global npm path
RUN npm install -g @earendil-works/pi-coding-agent

# Expose the port that the FastAPI service (knowledge_service) will run on
EXPOSE 8000

# Run the knowledge service using uvicorn
# We run it from the root so that 'knowledge_service' can be imported as a module
CMD ["uvicorn", "knowledge_service.main:app", "--host", "0.0.0.0", "--port", "8000"]

