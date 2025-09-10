# 🥦 CNN Vegetable Classifier  

A deep learning project for grayscale vegetable image classification, developed as part of the **ST1504 Deep Learning module** at Singapore Polytechnic. The project explores the effect of input resolution (**23×23 vs 101×101**) on CNN performance and integrates augmentation, balancing, and interpretability techniques.  

---

## 📌 Dataset  
- **Total Images**: 12,099 grayscale vegetable images  
- **Classes**: 11 (Beans, Bitter Gourd, Bottle Gourd & Cucumber, Brinjal, Broccoli & Cauliflower, Cabbage, Capsicum, Carrot & Radish, Potato, Pumpkin, Tomato)  
- **Original Size**: 224×224 RGB → converted to grayscale and resized to 23×23 and 101×101  
- **Cleaning**: Corrected mislabelled samples, unified inconsistent folder names, and removed noisy images  

---

## 🔑 Features  
- **Preprocessing & Augmentation**  
  - Grayscale conversion, resizing, and standardisation (0–1 scaling)  
  - Oversampling of minority classes for balance  
  - On-the-fly **MixUp augmentation** for stronger regularisation  

- **Model Architectures**  
  - Baseline CNN (3 Conv blocks + Dense)  
  - Balanced + Weighted CNNs to address class imbalance  
  - MixUp CNNs with BatchNorm, Dropout, LeakyReLU/ELU activations  
  - Compact Convolutional Transformer (CCT) hybrids  

- **Evaluation & Interpretability**  
  - Confusion matrices and F1-scores for balanced evaluation  
  - Training history plots for convergence monitoring  
  - **Grad-CAM** and **Occlusion Sensitivity** for model explainability  
  - Pixel intensity histograms, brightness/contrast analysis, and mean image visualisation  

---

## 🛠 Tech Stack  
- **Languages**: Python  
- **Libraries**: TensorFlow, Keras, NumPy, Pandas, OpenCV, Matplotlib, scikit-learn, Visualkeras  
- **Techniques**: CNNs, MixUp, Oversampling, Class Weighting, BatchNorm, Dropout, Hyperparameter Tuning, Transformers (CCT)  

---

## 📊 Results  
- **101×101 MixUp CNN (Tuned)** → 97.55% Test Accuracy | Strong interpretability via Grad-CAM  
- **23×23 CCT Hybrid (Tuned)** → 86.91% Test Accuracy | Robust despite low resolution  
- Oversampling + Class Weights improved fairness for minority classes  
- MixUp regularisation produced the most balanced and generalisable CNNs  
