## 🩺 Breast Cancer Detection

This project demonstrates how **machine learning** can be applied to predict whether a breast tumor is **benign** or **malignant** using the Breast Cancer Wisconsin dataset.  

The notebook walks through data loading, preprocessing, model training, evaluation, and building a simple predictive system. While this is primarily a **learning project**, it highlights the potential of data-driven approaches in healthcare research.  

---

### Dataset

- **Source**: [Breast Cancer Wisconsin dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset) (available via `sklearn.datasets.load_breast_cancer()`)  
- **Features**: 30 numerical features describing cell nuclei (e.g., radius, texture, smoothness)  
- **Target**:  
  - `0` → Malignant  
  - `1` → Benign  
- **Samples**: 569  

---

### Workflow


1. **Data Collection and Processing**  
   - Loaded dataset from scikit-learn  
   - Converted into a Pandas DataFrame  
   - Explored statistical measures and class distribution  

2. **Feature & Target Separation**  
   - Features (`X`) contain cell characteristics  
   - Target (`Y`) represents tumor class  

3. **Train-Test Split**  
   - Split dataset into training (80%) and testing (20%)  

4. **Model Training**  
   - Logistic Regression model trained on the training set  

5. **Model Evaluation**  
   - Accuracy measured on training and testing sets  
   - Model achieved high accuracy on both  

6. **Predictive System**  
   - Allows user to input feature values  
   - Predicts whether the tumor is benign or malignant  

---

### Technologies Used

- Python (>=3.8)  
- NumPy  
- Pandas  
- Scikit-learn  

---

### How to Run

1. **Clone the repository:**
    
    - git clone https://github.com/Shambhavi910/Breast_Cancer_Detection.git
    - cd Breast_Cancer_Detection


2. **Install dependencies:**

    - pip install -r requirements.txt


3. Open the Jupyter Notebook:

    - jupyter notebook Breast_cancer_detection.ipynb

### Results

Logistic Regression achieved high accuracy on both training and test datasets.

Predictive system successfully classifies new data inputs as malignant or benign.

### Future Enhancements

Experiment with additional models (Random Forest, SVM, Neural Networks).

Perform hyperparameter tuning for improved performance.

Deploy the predictive system using Flask or Streamlit.

### Disclaimer

This project is for educational purposes only.
It is not intended for medical or clinical use. Always consult healthcare professionals for diagnosis and treatment.
