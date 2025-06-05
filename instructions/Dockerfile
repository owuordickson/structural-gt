FROM ubuntu:latest
LABEL authors="owuor"

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y kubuntu-desktop xrdp && \
    adduser xrdp ssl-cert

RUN useradd -m owuor -p $(openssl passwd 1234) && \
    usermod aG sudo owuor

EXPOSE 3389

CMD service dbus start ; /usr/lib/systemd/systemd-logind & service xrdp start ; bash
