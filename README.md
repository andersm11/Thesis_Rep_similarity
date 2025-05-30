# Thesis Representation Similarity Analysis

This repository contains trained models and Centered Kernel Alignment (CKA) results for analyzing representation similarity in EEG-based tasks.

## 📁 Repository Structure

- `/FACED_models/`  
  Contains trained models and their CKA results for the FACED (emotion classification) dataset.

- `/motion_models/`  
  Contains trained models and CKA results for motor-related EEG classification tasks.

- `/Pipeline/python_models/CKA_functions.py`  
  Contains the implementation of the CKA functions used for representation similarity analysis.

- `/Pipeline/python_models/`  
  Contains the model architecture implementations used across experiments.

## ⚠️ Notes

- Some imports or file references may be broken due to recent refactoring and cleanup. Please ensure import paths are updated if you encounter issues when running scripts.

## 🧠 About CKA

Centered Kernel Alignment (CKA) is used here to measure the similarity of internal representations between different trained neural networks. This can provide insights into how different models or training settings affect learned features, even when model performance is similar.
