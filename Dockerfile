FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy all files and directories from the current directory into the container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your app runs on
EXPOSE 8080

# Run the Jupyter Notebook file as the entry point of the container
CMD ["python", "app.py", "--port","8080"]
