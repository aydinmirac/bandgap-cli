FROM --platform=linux/amd64 ubuntu:20.04

# Create a working directory
RUN mkdir /bangdap

# Update and install necessary programs
RUN apt update -y && apt upgrade -y

RUN apt install -y python3 python3-dev python3-pip wget curl ncurses-bin vim gcc

# Install necessary libraries
RUN pip3 install ase tqdm numpy

WORKDIR /bangdap
