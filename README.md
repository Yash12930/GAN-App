# GAN-App

A web-based application for generating images using Generative Adversarial Networks (GANs).

**Live Demo:** [https://gan-app.onrender.com/](https://gan-app.onrender.com/)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API](#api)
- [License](#license)

---

## Overview

**GAN-App** lets users generate images through a simple web interface powered by GANs. The backend is written in Python and exposes endpoints for image generation, while the frontend provides an easy way to interact with the model.

---

## Features

- Generate images using a trained GAN model.
- Simple, user-friendly web interface.
- RESTful API for programmatic access.
- Docker support for easy deployment.

---

## Project Structure
```
GAN-App/
├── api/ # Backend API (Python Flask)
├── app/ # Frontend (HTML/JS)
├── GAN.ipynb # Jupyter Notebook for GAN model development/training
├── requirements.txt # Python dependencies
├── Dockerfile # For containerized deployment
├── runtime.txt # Python runtime version
├── .gitignore
├── LICENSE
└── README.md
---
```
## Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

1. **Clone the repository:**
    ```
    git clone https://github.com/Yash12930/GAN-App.git
    cd GAN-App
    ```

2. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```

3. **Run the backend API:**
    ```
    cd api
    python app.py
    ```

4. **Open the frontend:**
    - Open `app/index.html` in your browser, or
    - Deploy both backend and frontend using the provided Dockerfile.

---

## Usage

- Visit the web interface to generate images.
- Use the provided API endpoints to generate images programmatically.

---

## API

The backend exposes endpoints for image generation. Example usage:

- **POST** `/generate`
    - **Body:** JSON with required parameters for image generation.
    - **Response:** Generated image (usually as a base64 string or image file).

*See `api/` for implementation details and example requests.*

---

## License

This project is licensed under the MIT License.

---

**Author:** [Yash12930](https://github.com/Yash12930)
