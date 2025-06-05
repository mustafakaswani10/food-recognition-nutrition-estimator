# 🍽️ Food Recognition & Nutrition Estimator

**End-to-end deep learning project:**  
AI-powered food image recognition (Food-101, PyTorch) with instant nutrition lookup, deployed as an interactive Streamlit app.

---

![Demo Screenshot](sample_screenshot.png)

---

## 🚀 Project Overview

- **Classifies food images** into 101 classes using a fine-tuned ResNet18 model.
- **Estimates nutrition facts** (calories, protein, carbs, fat) for each dish.
- **Interactive Streamlit app** for easy, real-time user testing.
- **Modular codebase**: Training notebooks, clean app logic, and a ready-to-edit nutrition database.

---

## ✨ Features

- **State-of-the-art food image classification** (Food-101 dataset)
- **Custom nutrition lookup** for each food class
- **Fast model training** with Google Colab (GPU)
- **Reproducible workflow:** Notebooks for data loading, EDA, and model training
- **No private or proprietary data required**

---

## 🗂️ Repository Structure
food-recognition-nutrition-estimator/
├── app/
│   ├── streamlit_app.py        # Streamlit app code
│   ├── model.pth               # Trained model weights
├── data/
│   ├── nutrition_lookup.csv    # Nutrition database (all 101 classes)
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb
│   ├── 02_training_colab.ipynb
├── requirements.txt
├── README.md
└── .gitignore

---

## ⚡ Quickstart

### 1. **Clone the repo**

```bash
git clone https://github.com/yourusername/food-recognition-nutrition-estimator.git
cd food-recognition-nutrition-estimator

pip install -r requirements.txt

3. Download Food-101 images (required for retraining)
	•	Download and extract Food-101
	•	Place the images/ and meta/ folders inside data/food-101/
	•	Do not add images to git! (They’re large—~5GB)

4. (Optional) Train your own model
	•	Open notebooks/02_training_colab.ipynb in Google Colab
	•	Follow instructions to train and export model.pth
	•	Place your trained model in app/model.pth

cd app

	•	Upload a food photo and get instant class prediction and nutrition info!
📊 How It Works
	•	Model: ResNet18, fine-tuned on 101 food classes
	•	Training: All training done on Google Colab (GPU recommended)
	•	Prediction: Fast, interactive web UI (Streamlit)
	•	Nutrition: Lookup table is editable (nutrition_lookup.csv), covers all 101 classes

⸻

🛠️ Customization
	•	Update nutrition info: Edit data/nutrition_lookup.csv for new or more accurate data
	•	Retrain model: Use your own images, new foods, or alternative architectures
	•	Expand app: Add portion sizing, batch processing, or user feedback as needed

⸻

📑 License

MIT — free for personal and commercial use.
Please cite this repo if you use it in research or a portfolio.
Enjoy using Food Recognition & Nutrition Estimator!
Feel free to open issues or contribute.
streamlit run streamlit_app.py
