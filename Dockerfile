ARG UBUNTU_VERSION=22.04
ARG NVIDIA_CUDA_VERSION=12.8.1

# Builder stage: installs needed dev packages and compiles dependencies
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as builder

# snoopy blackwell (120) + mila fleet: rtx8000 (75), a100 (80), a6000 (86), l40s (89), h100 (90)
ARG CUDA_ARCHITECTURES="75;80;86;89;90;120"
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages & dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        git \
        cmake \
        ninja-build build-essential \
        libboost-program-options-dev \
        libboost-graph-dev \
        libboost-system-dev \
        libeigen3-dev \
        libflann-dev \
        libfreeimage-dev \
        libmetis-dev \
        libgoogle-glog-dev \
        libgtest-dev \
        libgmock-dev \
        libsqlite3-dev \
        libglew-dev \
        qtbase5-dev \
        libqt5opengl5-dev \
        libcgal-dev \
        libceres-dev \
        libcurl4-openssl-dev \
        libgflags-dev \
        libatlas-base-dev \
        libsuitesparse-dev \
    && rm -rf /var/lib/apt/lists/*

# Build and install Ceres, Pyceres, Colmap, Pycolmap
RUN git clone --branch 2.2.0 --depth 1 https://ceres-solver.googlesource.com/ceres-solver /ceres-solver && \
    mkdir /ceres-solver/build && cd /ceres-solver/build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} -DBUILD_TESTING=OFF -DCMAKE_INSTALL_PREFIX=/usr/local && \
    ninja install
# pyceres v2.5: last release compatible with the colmap fork's pybind11 2.x
# bindings (pyceres main builds with pybind11 3.x, which cannot share
# ceres::Manifold types with pycolmap across modules)
RUN git clone --branch v2.5 --depth 1 https://github.com/cvg/pyceres.git /pyceres && \
    python3 -m pip install /pyceres
RUN python3 -m pip install ruff
ARG COLMAP_GIT_COMMIT=ab45bc79264cf6b521da9a350f4244ca10332511
RUN git clone https://github.com/adamburhan/colmap.git /colmap && \
    cd /colmap && git checkout ${COLMAP_GIT_COMMIT} && \
    mkdir build && cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} -DCMAKE_INSTALL_PREFIX=/usr/local && \
    ninja install && \
    python3 -m pip install /colmap

RUN apt-get update && apt-get install -y --no-install-recommends python3.10-venv && rm -rf /var/lib/apt/lists/*
ARG GS_COMMIT=4252af905a57b58ccfc8854b3225095aff2a3ce9
RUN python3 -m venv /opt/gs && \
    /opt/gs/bin/pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128 && \
    /opt/gs/bin/pip install wheel plyfile lpips joblib tqdm opencv-python-headless && \
    git clone --recursive https://github.com/adamburhan/gaussian-splatting.git /gaussian-splatting && \
    cd /gaussian-splatting && git checkout ${GS_COMMIT} && git submodule update --init --recursive && \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;12.0" /opt/gs/bin/pip install --no-build-isolation \
        submodules/diff-gaussian-rasterization submodules/simple-knn submodules/fused-ssim

ENV TORCH_HOME=/opt/torch-cache
# warm the caches for offline (cluster) use: vgg16 backbone via pip lpips, plus
# the LPIPS linear-head weights that gaussian-splatting's bundled lpipsPyTorch
# fetches through torch.hub at first call
RUN /opt/gs/bin/python -c "import lpips; lpips.LPIPS(net='vgg')" && \
    /opt/gs/bin/python -c "from torch.hub import load_state_dict_from_url; load_state_dict_from_url('https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth', map_location='cpu')"

# Slim stage: clean up dev files to reduce the size of the runtime COPY.
# Kept separate from builder so the dev stage retains headers and CMake configs.
FROM builder as builder-slim
RUN rm -rf \
    /colmap \
    /ceres-solver \
    /pyceres \
    /root/.cache \
    /usr/local/include \
    /usr/local/lib/cmake \
    /usr/local/share \
    /usr/local/lib/*.a \
    /usr/local/lib/*.la

# Dev stage: builder toolchain + Python deps for interactive C++/pycolmap iteration.
# Build with: docker build --target dev -t mpsfm-dev .
# Not used by the default (runtime) build.
FROM builder as dev
# runtime-only conveniences the builder stage lacks: wget (checkpoint
# downloads), 7z (benchmark dataset extraction) and the `python` alias
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget p7zip-full python-is-python3 && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
# mmcv installed separately with --no-build-isolation: its setup.py needs
# pkg_resources, removed in setuptools>=81, so it must build against our
# pinned setuptools instead of an isolated (latest) one. MMCV_WITH_OPS=0
# skips the CUDA ops, matching the upstream image (built without torch).
RUN python3 -m pip install --upgrade pip "setuptools<81" wheel && \
    grep -v -e 'ml-depth-pro' -e '^mmcv' /tmp/requirements.txt > /tmp/req.txt && \
    python3 -m pip install -r /tmp/req.txt && \
    MMCV_WITH_OPS=0 python3 -m pip install --no-build-isolation mmcv && \
    rm -rf /root/.cache && \
    # cholespy vendors metis.h/libmetis.a into the dist-packages prefix, which
    # scikit-build-core puts on CMAKE_PREFIX_PATH; that broken METIS poisons
    # SuiteSparse detection in any later pip-driven pycolmap/pyceres build
    rm -f /usr/local/lib/python3.10/dist-packages/include/metis.h \
          /usr/local/lib/python3.10/dist-packages/lib/libmetis.a
WORKDIR /mpsfm
ENTRYPOINT ["bash"]

# Runtime stage: minimal runtime dependencies
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} as runtime
ENV DEBIAN_FRONTEND=noninteractive

# Install only what's needed at runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip \
        python-is-python3 \
        git \
        wget \
        libboost-program-options-dev \
        libatlas-base-dev \
        libceres-dev \
        libfreeimage-dev \
        libglew-dev \
        libgoogle-glog-dev \
        libqt5core5a \
        libqt5gui5 \
        libqt5widgets5 \
        libcurl4 \
        # needed for compiling cuda kernels during runtime
        ninja-build \
        build-essential \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled artifacts from builder (slim variant, dev files stripped)
COPY --from=builder-slim /usr/local/ /usr/local/
ENV PATH=/usr/local/bin:$PATH

COPY --from=builder /opt/gs /opt/gs
ENV TORCH_HOME=/opt/torch-cache
COPY --from=builder /opt/torch-cache /opt/torch-cache
ENV PYTHONPATH=/mpsfm

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

WORKDIR /mpsfm
# Install Python requirements & finalize
COPY requirements.txt .
# mmcv handled separately: see dev stage note (needs pkg_resources at build time)
RUN python3 -m pip install --upgrade pip "setuptools<81" wheel && \
    grep -v -e 'ml-depth-pro' -e '^mmcv' requirements.txt > /tmp/req.txt && \
    pip install -r /tmp/req.txt && \
    MMCV_WITH_OPS=0 pip install --no-build-isolation mmcv && \
    rm -rf /root/.cache

# Final entrypoint
ENTRYPOINT ["bash"]
