FROM --platform=linux/amd64 ubuntu:22.04

# Create a working directory
RUN mkdir /bandgap

# Update and install necessary programs
RUN apt update -y && apt upgrade -y

RUN apt install -y python3 python3-dev python3-pip wget curl ncurses-bin vim gcc

COPY auglichem/ /bandgap/auglichem
COPY data-import/ /bandgap/data-import
COPY functions.py main.py /bandgap/

WORKDIR /bandgap
