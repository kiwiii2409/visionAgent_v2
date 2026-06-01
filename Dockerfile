# visionagent Docker image
# Based on linuxserver/webtop:ubuntu-xfce for an isolated XFCE desktop
FROM lscr.io/linuxserver/webtop:ubuntu-xfce

# System dependencies for visionagent (pyautogui, mss, Chroma, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libx11-dev \
    libxtst-dev \
    libpng-dev \
    scrot \
    xdotool \
    && rm -rf /var/lib/apt/lists/*

# Copy visionagent code
COPY --chown=abc:abc . /app/visionagent

WORKDIR /app/visionagent

# Install Python dependencies
RUN pip3 install --no-cache-dir --break-system-packages -e .

# s6-overlay service: start FastAPI after desktop is ready
COPY docker/visionagent-service/run /etc/services.d/visionagent/run
RUN chmod +x /etc/services.d/visionagent/run

# Force X11 (pyautogui/mss need Xlib, not Wayland)
ENV DISPLAY=:1 \
    PIXELFLUX_WAYLAND=false \
    DOCKER_MODE=true

EXPOSE 8000 3000
