# Use RHEL Universal Base Image (UBI) 9
ARG RHEL_VERSION=9.3
FROM registry.access.redhat.com/ubi9/ubi:9.3

# Arguments for CUDA version
ARG CUDA_VERSION=12.3.1
ARG CUDA_MAJOR_VERSION=${CUDA_VERSION%.*.*}

# Install required packages and perform system update
RUN dnf -y update && \
    dnf -y upgrade && \
    dnf -y install \
    dnf-plugins-core \
    epel-release \
    wget \
    which \
    python3.11 \
    python3.11-pip \
    python3.11-devel \
    gcc \
    gcc-c++ \
    make \
    git \
    numactl \
    libnuma-devel \
    hwloc \
    hwloc-devel \
    && dnf clean all \
    && rm -rf /var/cache/dnf/*

# Set up CUDA repository and install CUDA
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo -O /etc/yum.repos.d/cuda-rhel9.repo && \
    dnf -y module install nvidia-driver:latest-dkms && \
    dnf -y install \
    cuda-${CUDA_MAJOR_VERSION}.x86_64 \
    cuda-devel-${CUDA_MAJOR_VERSION}.x86_64 \
    cuda-nvml-devel-${CUDA_MAJOR_VERSION}.x86_64 \
    cuda-command-line-tools-${CUDA_MAJOR_VERSION}.x86_64 \
    && dnf clean all && \
    rm -rf /var/cache/dnf/*

# Set up environment variables
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# NVIDIA DALI and GPU Performance Optimizations
ENV DALI_USE_DEVICE_MEM_POOL=1
ENV DALI_USE_CUDA_MALLOC_ASYNC=1
ENV DALI_USE_VMM=1
ENV DALI_GDS_CHUNK_SIZE=2M
ENV DALI_ODIRECT_ALIGNMENT=4K
ENV DALI_ODIRECT_LEN_ALIGNMENT=4K
ENV DALI_ODIRECT_CHUNK_SIZE=2M
ENV DALI_PREFETCH_QUEUE_DEPTH=3

# GPU Compute Optimization Environment Variables
ENV CUDA_DEVICE_MAX_CONNECTIONS=32
ENV CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
ENV CUDA_AUTO_BOOST=0
ENV CUDA_VISIBLE_DEVICES=all
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
ENV NVIDIA_REQUIRE_CUDA="cuda>=${CUDA_MAJOR_VERSION}.0"

# Install CUDA Python tools
RUN python3.11 -m pip install --no-cache-dir \
    numpy \
    cupy-cuda${CUDA_MAJOR_VERSION}x \
    numba \
    nvidia-dali-cuda${CUDA_MAJOR_VERSION} \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu${CUDA_MAJOR_VERSION}

# Create a non-root user named lenny
RUN useradd -m -s /bin/bash lenny && \
    echo "lenny ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Create necessary directories and set permissions
RUN mkdir -p /workspace /scratch /cache && \
    chown -R lenny:lenny /workspace /scratch /cache

# Switch to non-root user
USER lenny
WORKDIR /workspace

# Configure GPU compute settings
RUN mkdir -p ~/.config/nvidia && \
    echo "{ \
        \"version\": \"1.0\", \
        \"threads\": $(nproc), \
        \"device_filter\": \"all\", \
        \"device_sort\": \"memory\", \
        \"gpu_memory_fraction\": 0.95, \
        \"unified_memory\": true, \
        \"cuda_cache_path\": \"/cache/cuda\", \
        \"cuda_cache_maxsize\": \"2G\" \
    }" > ~/.config/nvidia/nvidia-settings.json

# Add container metadata
LABEL org.opencontainers.image.source=https://github.com/${GITHUB_REPOSITORY}
LABEL org.opencontainers.image.description="RHEL ${RHEL_VERSION} with CUDA ${CUDA_VERSION} and Python tools"
LABEL org.opencontainers.image.licenses="MIT"
LABEL cuda.version="${CUDA_VERSION}"
LABEL rhel.version="${RHEL_VERSION}"
LABEL nvidia.optimizations.dali="enabled"
LABEL nvidia.optimizations.rendering="enabled"

# Initialize DALI memory pools (helps avoid runtime performance drops)
RUN python3.11 -c 'from nvidia.dali.backend import PreallocateDeviceMemory, PreallocatePinnedMemory; \
    import torch; \
    device = torch.cuda.current_device(); \
    PreallocateDeviceMemory(1024*1024*1024, device); \
    PreallocatePinnedMemory(256*1024*1024)'

# Default command
CMD ["python3.11"] 