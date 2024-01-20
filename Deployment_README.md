# Deployment Details

In this section, we will discuss detailed steps on how to deploy the model using FastAPI, Docker, Azure, and GitHub Actions.

## Prerequisites

Before starting the deployment process, make sure you have the following:

1. Azure account: Create an account on the [Azure portal](https://portal.azure.com/).
2. Docker: Install Docker on your local machine. You can download it from the [Docker website](https://www.docker.com/get-started).
3. GitHub account: Create a GitHub account if you don't have one.

## Steps to Deploy the Model

### 1. Create a Dockerfile

Create a file named `Dockerfile` in the root directory of your project. This file contains instructions for building a Docker image.

```Dockerfile
FROM python:3.7-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
```

### 2. Create a requirements.txt file

Create a file named `requirements.txt` in the root directory. This file should contain the required Python packages for your project.

```python
pip freeze
```
- Save all the packages in the requirements.txt file.

```plaintext
fastapi
uvicorn
torch
# Add other dependencies as needed
```

### 3. Create a FastAPI App

- Create a FastAPI application in a file named `app.py`. 
- This file should import the generator.pth model exported from the ipynb file.
- Write code to load the model and generate images using the model.

### 4. Build Docker Image

- Open a terminal in the project directory and run the following command to build the Docker image:

```bash
docker build -t name_of_image:latest .
```

### 5. Test the Docker Container Locally

- Run the following command to start the Docker container locally:
```bash
docker run -p 5000:5000 _image_name_
```
- Visit [http://localhost:5000](http://localhost:5000) in your browser to test the FastAPI app.
```bash
docker run -p 4000:80 my-fastapi-app
```
- Visit [http://localhost:4000](http://localhost:4000) in your browser to test the FastAPI app.

### 6. Create a GitHub Repository

- Create a new repository on GitHub to host your project.

### 7. Create a GitHub Action
- This step depends, Azure creates a github action for you or else you can create a github action mannualy.
- Create a file named `.github/workflows/main.yml` in your project repository. This file defines the GitHub Actions workflow.

```yaml
name: CI/CD with Docker

on:
  push:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v2

    - name: Build Docker image
      run: docker build -t my-fastapi-app .

    - name: Push Docker image to GitHub Container Registry
      run: echo ${{ secrets.GITHUB_TOKEN }} | docker login ghcr.io -u ${{ github.actor }} --password-stdin
      - run: docker tag my-fastapi-app ghcr.io/your-username/my-fastapi-app:latest
      - run: docker push ghcr.io/your-username/my-fastapi-app:latest
```

- Replace `your-username` with your GitHub username.

### 8. Create an Azure Container Registry
- Follow the images to deploy the model on Azure.
1. ![image](/output_images/Azure/azure-01.png)
2. ![image](/output_images/Azure/azure-02.png)
3. ![image](/output_images/Azure/azure-03.png)
4. ![image](/output_images/Azure/azure-04.png)
5. ![image](/output_images/Azure/azure-05.png)

- Get login and pass from access
	Login Server: dcganime.azurecr.io
	Username: dcganime
	pass: **********************************************
```bash
docker login _login_from_azure_
```

### 9. Update GitHub Action for Azure Deployment

Update the `.github/workflows/main.yml` file to include the Azure deployment steps:

```yaml
name: CI/CD with Docker

on:
  push:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v2

    - name: Build Docker image
      run: docker build -t my-fastapi-app .

    - name: Push Docker image to GitHub Container Registry
      run: echo ${{ secrets.GITHUB_TOKEN }} | docker login ghcr.io -u ${{ github.actor }} --password-stdin
      - run: docker tag my-fastapi-app ghcr.io/your-username/my-fastapi-app:latest
      - run: docker push ghcr.io/your-username/my-fastapi-app:latest

  deploy:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v2

    - name: Set up Azure CLI
      uses: azure/setup-azure-cli@v1
      with:
        azcliversion: 2.0.72

    - name: Log in to Azure Container Registry
      run: az acr login --name your-acr-name.azurecr.io --username your-acr-username --password your-acr-password

    - name: Pull Docker image from GitHub Container Registry
      run: docker pull ghcr.io/your-username/my-fastapi-app:latest

    - name: Tag Docker image for Azure Container Registry
      run: docker tag ghcr.io/your-username/my-fastapi-app:latest your-acr-name.azurecr.io/my-fastapi-app:latest

    - name: Push Docker image to Azure Container Registry
      run: docker push your-acr-name.azurecr.io/my-fastapi-app:latest
```

Replace `your-acr-name`, `your-acr-username`, and `your-acr-password` with the appropriate values from your Azure Container Registry.

### 10. Create an Azure Web App
1. In the Azure portal, navigate to "App Services" and create a new Web App.

2. Configure the Web App settings, and under "Container Settings," select the Azure Container Registry and specify the image to deploy.

![image](/output_images/Azure/azure-06.png)
![image](/output_images/Azure/azure-07.png)
![image](/output_images/Azure/azure-08.png)
![image](/output_images/Azure/azure-09.png)

### 11. Push Changes to GitHub

Commit and push the changes to your GitHub repository.

### 12. Monitor GitHub Actions Workflow

Check the GitHub Actions workflow in the "Actions" tab of your repository on GitHub. It should trigger automatically on each push to the main branch.

### 13. Access the Deployed FastAPI App

Once the workflow is successful, access your deployed FastAPI app by visiting the Azure Web App URL.
![image](/output_images/Azure/azure-10.png)
![image](/output_images/Azure/azure-11.png)

> Note: Same steps can be followed to deploy the model on AWS, Google Cloud, or any other cloud platform.