FROM ubuntu:20.04

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies for python and ViZDoom
RUN apt-get update && apt-get install -y \
    build-essential \
    bzip2 \
    cmake \
    curl \
    git \
    libbz2-dev \
    libfluidsynth-dev \
    libgme-dev \
    libgtk2.0-dev \
    libjpeg-dev \
    libopenal-dev \
    libsdl2-dev \
    libwildmidi-dev \
    libz-dev \
    tar \
    unzip \
    wget \
    zlib1g-dev \
    python3.9 \
    python3.9-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set python3.9 as default
RUN ln -sf /usr/bin/python3.9 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python requirements
RUN pip install -r requirements.txt

# Copy the entire project code base into the container
COPY . .

# Set the default command to train the basic scenario
ENTRYPOINT ["python", "-m", "src.train"]
CMD ["--scenario", "basic", "--timesteps", "100000"]
