# 🍽️ Food Recognition & Nutrition Estimator

End-to-end deep learning project:  
AI-powered food image recognition (Food-101, PyTorch) with instant nutrition lookup, deployed as an interactive Streamlit app.

---

## 🟢 Try the App Online

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://food-estimator.streamlit.app/)

Or click here: https://food-estimator.streamlit.app/

(![Screenshot 2025-06-05 at 7 29 09 PM](https://github.com/user-attachments/assets/59010c08-bed5-4401-8378-04b136fddd34)
)

---

## 🚀 Project Overview

- **Classifies food images** into 101 classes using a fine-tuned ResNet18 model.
- **Estimates nutrition facts** (calories, protein, carbs, fat) for each dish.
- **Interactive Streamlit app** for easy, real-time user testing.
- **Modular codebase:** Training notebooks, clean app logic, and an editable nutrition database.

---

## 🗂️ Repository Structure
food-recognition-nutrition-estimator/

<pre>
```
food-recognition-nutrition-estimator/
├── app/
│   ├── streamlit_app.py
│   ├── model.pth
├── data/
│   ├── nutrition_lookup.csv
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb
│   ├── 02_training_colab.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```
</pre>

---

## ⚡ Quickstart

1. **Clone the repository**
    ```bash
    git clone https://github.com/mustafakaswani10/food-recognition-nutrition-estimator.git
    cd food-recognition-nutrition-estimator
    ```

2. **Install requirements**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download Food-101 images (for retraining)**
    - Download and extract [Food-101](https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)
    - Place the `images/` and `meta/` folders inside `data/food-101/`
    - **Do not add images to git!** (They're large—~5GB)

4. **(Optional) Train your own model**
    - Open `notebooks/02_training_colab.ipynb` in [Google Colab](https://colab.research.google.com/)
    - Follow instructions to train and export `model.pth`
    - Place your trained model in `app/model.pth`

5. **Run the Streamlit app**
    ```bash
    cd app
    streamlit run streamlit_app.py
    ```
    - Upload a food photo and get instant class prediction and nutrition info!

---

## 📊 How It Works

- **Model:** ResNet18, fine-tuned on 101 food classes (Food-101 dataset)
- **Training:** All training done on Google Colab (GPU recommended)
- **Prediction:** Fast, interactive web UI (Streamlit)
- **Nutrition:** Lookup table is editable (`nutrition_lookup.csv`), covers all 101 classes

---

## 🛠️ Customization

- Update nutrition info: Edit `data/nutrition_lookup.csv` for new or more accurate data
- Retrain model: Use your own images, new foods, or alternative architectures
- Expand app: Add portion sizing, batch processing, or user feedback as needed

---

## 📑 License

MIT — free for personal and commercial use.  
Please cite this repo if you use it in research or a portfolio.

---

## 🙋‍♂️ Contact

[Mustafa Kaswani](mailto:mustafa.hkaswani@gmail.com)  
[LinkedIn](https://www.linkedin.com/in/mustafahussainkaswani/)

---

**Enjoy using Food Recognition & Nutrition Estimator! Feel free to open issues or contribute.**
