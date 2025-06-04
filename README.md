# CUDA Container Images for RHEL 9

This repository contains container images based on Red Hat Enterprise Linux (RHEL) 9, optimized for NVIDIA GPUs with CUDA support. The images include Python tools and libraries commonly used in GPU computing and rendering workloads.

## Available Images

The following CUDA versions are supported:
- CUDA 12.3.1
- CUDA 12.2.2
- CUDA 12.1.1
- CUDA 11.8.0

Each image is based on RHEL 9.3 UBI (Universal Base Image) and includes:
- NVIDIA GPU Drivers (latest DKMS version)
- CUDA Toolkit and Development Tools
- CUDA NVML Development Tools
- CUDA Command Line Tools
- Python 3.11 with GPU-enabled packages:
  - NumPy
  - CuPy
  - Numba
  - NVIDIA DALI (Data Loading Library)
  - PyTorch (with CUDA support)
- Non-root user 'lenny' with sudo privileges
- Latest system updates and security patches
- Hardware locality (hwloc) tools for NUMA optimization
- NUMA development libraries

## Performance Optimizations

The containers include several optimizations for GPU computing and rendering:

### Memory Management
- Pre-allocated GPU device memory (1GB)
- Pre-allocated pinned host memory (256MB)
- Enabled CUDA memory pools
- Optimized Virtual Memory Management (VMM)
- NUMA-aware memory allocation
- Hardware topology optimization via hwloc

### GPU Compute Settings
- Maximum CUDA device connections (32)
- Forced device memory allocation for managed memory
- Disabled auto-boost for consistent performance
- Unified memory support
- GPU memory fraction set to 95%
- 2GB CUDA cache in /cache/cuda
- Memory-based device sorting

### I/O Performance
- Optimized GDS chunk size (2MB)
- Configured O_DIRECT alignments for better I/O performance
- Increased prefetch queue depth for smoother batch processing
- Dedicated scratch and cache directories

### Environment Variables
The following performance-related environment variables are pre-configured:

```bash
# DALI Optimizations
DALI_USE_DEVICE_MEM_POOL=1
DALI_USE_CUDA_MALLOC_ASYNC=1
DALI_USE_VMM=1
DALI_GDS_CHUNK_SIZE=2M
DALI_ODIRECT_ALIGNMENT=4K
DALI_ODIRECT_LEN_ALIGNMENT=4K
DALI_ODIRECT_CHUNK_SIZE=2M
DALI_PREFETCH_QUEUE_DEPTH=3

# CUDA Optimizations
CUDA_DEVICE_MAX_CONNECTIONS=32
CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
CUDA_AUTO_BOOST=0
CUDA_VISIBLE_DEVICES=all
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
```

## Image Tags

Images are tagged using the following formats:
- `cuda<version>` (e.g., `cuda12.3.1`)
- `cuda<version>-rhel<version>` (e.g., `cuda12.3.1-rhel9.3`)
- `v<release>-cuda<version>` (e.g., `v1.0.0-cuda12.3.1`)
- `v<release>-cuda<version>-rhel<version>` (e.g., `v1.0.0-cuda12.3.1-rhel9.3`)

## Using the Images

Pull the desired image from the GitHub Container Registry:

```bash
# Pull by CUDA version
docker pull ghcr.io/OWNER/REPOSITORY:cuda12.3.1

# Pull by CUDA and RHEL version
docker pull ghcr.io/OWNER/REPOSITORY:cuda12.3.1-rhel9.3

# Pull specific release
docker pull ghcr.io/OWNER/REPOSITORY:v1.0.0-cuda12.3.1
```

Replace `OWNER/REPOSITORY` with your GitHub repository path.

### Running the Container

To run the container with GPU support:

```bash
docker run --gpus all \
  --cap-add=SYS_NICE \
  --security-opt seccomp=unconfined \
  -it ghcr.io/OWNER/REPOSITORY:cuda12.3.1
```

The container runs as non-root user 'lenny' by default, with access to:
- `/workspace` directory (default working directory)
- `/scratch` directory for temporary files
- `/cache` directory for CUDA cache
- sudo privileges for system administration tasks
- CUDA toolkit and development tools
- Pre-configured NVIDIA DALI optimizations
- GPU compute optimizations

### Environment Variables

The following environment variables are pre-configured:
- `PATH`: Includes CUDA binary path
- `LD_LIBRARY_PATH`: Includes CUDA library path
- Various DALI performance optimization variables
- CUDA and NVIDIA optimization variables

## Building Custom Images

To build a custom image locally:

```bash
docker build \
  --build-arg CUDA_VERSION=12.3.1 \
  --build-arg RHEL_VERSION=9.3 \
  -t my-cuda-image .
```

## GitHub Actions Workflow

The images are automatically built and published using GitHub Actions when:
1. A new tag is pushed (format: `v*`) - builds all CUDA versions
2. The workflow is manually triggered (with custom CUDA and RHEL version selection)

## Security

- Images run as non-root user 'lenny'
- Latest security updates are included
- System packages are cleaned up to reduce image size
- NVIDIA drivers are installed from official repositories
- Memory management is optimized for security and performance
- NUMA security boundaries are respected

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 