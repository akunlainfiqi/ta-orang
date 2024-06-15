# Use an official Python runtime as a parent image
FROM --platform=linux/amd64 python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container to /app
WORKDIR /app

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

ENV BUILD_DEPENDENCIES="build-essential"
RUN apt-get update \
    && apt-get install -y $BUILD_DEPENDENCIES \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get remove -y $BUILD_DEPENDENCIES \
    && apt-get auto-remove -y \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run the application:
CMD ["flask", "run", "--host=0.0.0.0"]