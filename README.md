# **VITE: Visualization Tool for Tree Ensembles**

This repository contains the implementation of **VITE (Visualization Tool for Tree Ensembles)**, as presented in the research paper *"Unboxing Tree Ensembles for Interpretability: A Hierarchical Visualization Tool and a Multivariate Optimal Re-Built Tree"* by Giulia Di Teodoro, Marta Monaci, and Laura Palagi.

## **Overview**

VITE is a hierarchical visualization tool designed to interpret and analyze Tree Ensemble (TE) models like Random Forests and XGBoost. It focuses on enhancing the interpretability of tree ensembles, which are often considered "black-box" models due to their complexity. 

### **Key Features of VITE:**
1. **Feature-Level Visualization**: A heatmap-based tool to highlight feature usage at various levels of the tree ensemble.
2. **Proximity Matrix**: Computes proximity measures among samples to analyze cluster formations within the tree ensemble.
3. **Hierarchical Visualization**: Provides insights into the importance of features across multiple trees in the ensemble.
4. **Support for Optimal Representer Trees**: VITE serves as a step towards building interpretable surrogate models (e.g., Multivariate Interpretable Re-built Trees - MIRET).

## **Purpose**

The provided code implements the "VITE" component, focusing on:
- Visualizing feature usage levels across tree depths using heatmaps.
- Understanding how features impact decisions in the ensemble.
- Generating data structures (e.g., proximity matrices) to aid in surrogate modeling.

The code specifically covers the **hierarchical visualization** of feature usage and proximity-based sample clustering within tree ensembles.

## **Code Structure**
- `main_xgb.py`: Main script to load data, train XGBoost models, and generate visualizations using the VITE tool.
- `training.py`: Contains helper functions to preprocess data, train models, and evaluate performance.
- `utils_xgb.py`: Utility functions to compute proximity matrices, feature importance, and hierarchical visualizations.

## **How to Use**
### **Prerequisites**
- Install required Python packages:
  ```bash
  pip install numpy pandas matplotlib seaborn xgboost scikit-learn imbalanced-learn
