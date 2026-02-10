# Patent License Prediction Using Deep Survival Analysis
This repository contains the official implementation for the paper:

"Patent License Prediction Using Deep Survival Analysis: A Comparative Study"

## Overview
This study evaluates deep survival analysis frameworks for predicting patent licensing events. We systematically compare neural network-based survival models (DeepSurv) and cure rate models against classical Cox regression and static classification approaches. Using a large-scale dataset of over 600,000 USPTO patents, we demonstrate the superiority of temporal modeling in patent transaction contexts.

## Key Features
Implementation of DeepSurv and Cox Cure Rate Models for patent data.

Comparative analysis using Recall@K metrics across multiple time horizons (1, 3, and 5 years).

 Pre-processing scripts for USPTO patent assignment data.

Evaluation of temporal modeling vs. static deep learning classification.

## Data Availability
The analysis utilizes the USPTO Patent Assignment Dataset. Due to file size constraints, raw data is not hosted in this repository. Please refer to the data/ directory for scripts to fetch and prepare the dataset from the Google Patents Public Data (BigQuery).
