# Pull tensorflow image with Python3
FROM tensorflow/tensorflow:2.18.0

# Set the working directory to /app
WORKDIR /app

# Transfer content from current dir to /app in container
ADD . /app

# Install audio libraries
# RUN apt-get update && apt-get install -y libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg

# Install python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --ignore-installed --no-cache-dir -r requirements.txt

# Start Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5050", "server:app"]