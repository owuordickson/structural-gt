FROM ubuntu:latest
LABEL authors="owuor"

# Avoid interactive prompts during install
ENV DEBIAN_FRONTEND=noninteractive

# Install desktop environment and XRDP
RUN apt update && \
    apt install -y kubuntu-desktop xrdp sudo openssl && \
    adduser xrdp ssl-cert

# Create a new user with password and add to sudo
RUN useradd -m -s /bin/bash owuor && \
    echo "owuor:1234" | chpasswd && \
    usermod -aG sudo owuor

# Expose XRDP port
EXPOSE 3389

# Start services
CMD ["/bin/bash", "-c", "service dbus start && service xrdp start && tail -f /dev/null"]
