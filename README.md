# Amazon ML Challenge 2024

## Overview

This repository presents the solution for the Amazon ML Challenge 2024. The challenge's objective was to develop a machine learning model capable of extracting entity values from product images, a task that holds significance in domains such as e-commerce, healthcare, and content moderation. As digital marketplaces grow, extracting detailed textual descriptions from images becomes increasingly important, providing crucial information like weight, volume, voltage, wattage, and dimensions.

## Model Approach

The solution was developed using a Vision-Language Model (VLM) named 'MoonDream2,' available on HuggingFace ([MoonDream2 Model](https://huggingface.co/vikhyatk/moondream2)). The approach involved the following steps:

### Step 1: Initial Testing of VLM

Initially, the pre-trained Vision-Language Model (VLM) was tested with a few images and prompts to assess its performance. The model's initial results were not optimal, but they indicated potential for improvement through fine-tuning.

### Step 2: Fine-Tuning the Language Part of the Model

The pre-trained model's outputs were lengthy and misaligned with the desired results. Therefore, fine-tuning focused on the language component of the model. The model weights were downloaded from HuggingFace, and fine-tuning was performed using 300 batches of training data with a batch size of 8 on a T4 GPU provided by Kaggle Notebooks. This initial fine-tuning yielded an F1 score of 45.2%. An exploratory data analysis (EDA) on the test data revealed inconsistencies, necessitating further data cleaning.

### Step 3: Extensive Fine-Tuning with Data Cleaning

Recognizing the importance of fine-tuning, the model was subjected to extensive fine-tuning for 2250 batches (~35,000 images) with a batch size of 16 using Kaggle's GPU. The data underwent thorough cleaning to ensure compliance with the guidelines, such as replacing 'horsepower' with 'blank' when the entity was not present in the image and removing problematic URLs. This process resulted in a significant improvement, raising the F1 score to 61.2%. Additionally, optimizing the prompt further enhanced the model's effectiveness.

### Step 4: Further Fine-Tuning with Full Training Data

Given the positive impact of extensive fine-tuning, the model was fine-tuned using the entire training dataset. With A100 GPUs from Google Colab Pro, the model was further fine-tuned using a batch size of 16. Although fine-tuning the entire dataset was not possible due to time constraints, an additional 10,000 iterations were completed, totaling 2250 + 10,000 iterations/batches (~196,000 images). The final F1 score obtained was 63.8%.

### Prompt Used
The prompt used for extracting entities from the images was:
```
Extract (entity name without ‘_’) from the image in the format 'x unit', where: 'x' is a float number in standard formatting. 'unit' is one of the allowed units for {entity name without ‘’} from the following list: (corresponding units are taken from entity_unit_map). Ensure that the output strictly matches the format "x unit" with a space separating the number and the unit. Do not use any abbreviations, special characters, or additional text. If no valid value is found in the image, return a string "blank".
```

## Results

The extensive fine-tuning, combined with data cleaning and prompt optimization, led to a significant improvement in the model's performance, achieving an F1 score of 63.8%.

## Model and Training Details

- **Model**: MoonDream2 - Vision-Language Model available on [HuggingFace](https://huggingface.co/vikhyatk/moondream2)
- **Training Environment**: Kaggle Notebooks with T4 GPU, Google Colab Pro with A100 GPU
- **Training Data**: Approximately 196,000 images were used after extensive cleaning and preprocessing.
