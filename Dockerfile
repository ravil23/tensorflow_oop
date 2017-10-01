# Use an official Python runtime as a parent image
FROM python:2.7-slim

# Export env settings
ENV LANG en_US.UTF-8
ENV LC_ALL C

# Set the working directory to /tensorflow_oop
WORKDIR /tensorflow_oop

# Copy the current directory contents into the container at /tensorflow_oop
ADD . /tensorflow_oop

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
RUN python setup.py install

# Run tests
RUN python test/test_bag_of_words.py
RUN python test/test_dataset.py
RUN python test/test_sequence.py
RUN python test/test_tripletset.py

# Define environment variable
ENV NAME "TensorFlow OOP"

# Run bash when the container launches
CMD ["/bin/bash"]