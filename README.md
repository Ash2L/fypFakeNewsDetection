

# Fake News Detector

Welcome to the Fake News Detector project! This web application is designed to detect fake textual contents using machine learning algorithms.

## Introduction
The Fake News Detector is a Flask-based web application that utilizes machine learning models to classify news as either real or fake. The application provides a simple user interface where users can input the title and content of a news article, and it will predict whether the article is genuine or not.

## Features
- User-friendly interface for submitting news articles
- Machine learning model integration for accurate fake news detection
- Real-time prediction of news article authenticity

## Installation
To run the Fake News Detector locally, follow these steps:

1. Clone the repository:

2. Pip install flask, torch and transformer

3. Open the integrated terminal and type flask run

If you face unpickling error, it is due to the size of the model being too large and it being incorrectly downloaded or it being corrupted.
If this error persist, you can download the model called "engDetect.pt" from here: https://drive.google.com/drive/folders/1u2eHG85kxHSiVtqxTq7GI3Wkf8TSVkg0?usp=sharing
You can also download the fake news detection file from the google docs link if any error occurs.
