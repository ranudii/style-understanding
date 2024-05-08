# Style Understanding Project

This project aims to classify images of interior design styles using machine learning. The application allows users to upload images and receive a classification result indicating the dominant design style.

## Project Structure

The project contains the following key components:
- **`scrape_image.py`**: Script to scrape images from Google Images for training purposes.
- **`train_inception.py`**: Script to train the classification model.
- **`ModelClass.py`**: Defines the model used for inference.
- **`dedup_image.py`**: Script to remove duplicate images from the training dataset.
- **`app.py`**: The main application server script using Flask.
- **`Interior_design.html`**: The HTML file providing the user interface.
- **`static`**: Contains the CSS and JavaScript files.

## Installation and Setup

### Prerequisites
- Python 3.6+
- Virtual Environment (optional but recommended)


### Running the Web Application
1. **Flask Application**:
    - Run the Flask application with `app.py`:
    ```bash
    python app.py
    ```
    - Access the application via `http://localhost:5000`.


### Dataset Preparation
1. **Scraping Images**:
    - Run `scrape_image.py` to collect images for training.
    - Images are labeled according to search terms.

2. **Removing Duplicates**:
    - Execute `dedup_image.py` to clean the dataset.

### Training the Model
1. **Training**:
    - Run `train_inception.py` to train the classification model.
    - The final model will be saved as `inV3_last_layer.h5`.


