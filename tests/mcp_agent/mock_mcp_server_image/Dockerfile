# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY mock_server.py .

# Make port 8080 available to the world outside this container
# The actual port used by the server can be configured by MOCK_MCP_PORT env var
EXPOSE 8080

# Define environment variable for the default port if not overridden
ENV MOCK_MCP_PORT=8080
ENV MOCK_SERVER_ID="docker_mock_server"

# Run mock_server.py when the container launches
CMD ["python", "mock_server.py"]
