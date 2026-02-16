FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
# This line tells FFmpeg where the NVIDIA header metadata (ffnvcodec.pc) is located
ENV PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/local/share/pkgconfig:${PKG_CONFIG_PATH}"
ENV DEBIAN_FRONTEND=noninteractive

# 1. Added pkg-config (required) and nasm (better than yasm for modern builds)
RUN apt-get update && apt-get install -y \
    build-essential yasm nasm cmake libtool libc6 libc6-dev unzip wget libnuma1 libnuma-dev git ninja-build meson \
    libdav1d-dev libopus-dev libfdk-aac-dev libvpx-dev libx265-dev libx264-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/build

# 2. Install NVIDIA headers
RUN git clone https://github.com/FFmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && make install

# 3. Build libvmaf
RUN git clone --depth 1 https://github.com/Netflix/vmaf.git && \
    cd vmaf/libvmaf && \
    meson setup build \
      --buildtype release \
      --prefix=/usr/local \
      -Dbuilt_in_models=true && \
    ninja -vC build install && \
    ldconfig

WORKDIR /opt/build/FFmpeg-src
RUN git clone --depth 1 https://github.com/FFmpeg/FFmpeg.git .

# 4. Build FFmpeg
# Added --enable-libvmaf and --enable-gpl (required for x264/vmaf)
RUN ./configure \
    --enable-nonfree \
    --enable-gpl \
    --enable-cuda-nvcc \
    --extra-cflags=-I/usr/local/cuda/include \
    --extra-ldflags=-L/usr/local/cuda/lib64 \
    --enable-nvenc \
    --enable-nvdec \
    --enable-libvmaf \
    --enable-libx264 \
    --enable-libx265 \
    --disable-static \
    --enable-shared && \
    make -j$(nproc) && \
    make install && \
    ldconfig

WORKDIR /video

# Simple bash entry for easy exec
CMD ["tail", "-f", "/dev/null"]