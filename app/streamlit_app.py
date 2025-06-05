import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import pandas as pd
import os

# --- Configs ---
food_classes = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad", "beignets",
    "bibimbap", "bread_pudding", "breakfast_burrito", "bruschetta", "caesar_salad", "cannoli", "caprese_salad",
    "carrot_cake", "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder", "club_sandwich",
    "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes", "deviled_eggs", "donuts", "dumplings",
    "edamame", "eggs_benedict", "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt",
    "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich", "grilled_salmon", "guacamole",
    "gyoza", "hamburger", "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream",
    "lasagna", "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters", "pad_thai", "paella", "pancakes", "panna_cotta",
    "peking_duck", "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich", "ramen",
    "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake", "sushi",
    "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles"
]
NUTRITION_DF = pd.read_csv("../data/nutrition_lookup.csv")
IMG_SIZE = 224

# --- Load model ---
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(FOOD_CLASSES))
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# --- Image preprocessing ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Nutrition lookup function ---
def lookup_nutrition(food_class):
    row = NUTRITION_DF[NUTRITION_DF["food"] == food_class]
    if len(row) > 0:
        return {
            "Calories (kcal)": int(row.iloc[0]["calories_kcal"]),
            "Protein (g)": int(row.iloc[0]["protein_g"]),
            "Carbs (g)": int(row.iloc[0]["carbs_g"]),
            "Fat (g)": int(row.iloc[0]["fat_g"])
        }
    else:
        return {
            "Calories (kcal)": "unknown",
            "Protein (g)": "unknown",
            "Carbs (g)": "unknown",
            "Fat (g)": "unknown"
        }

# --- Streamlit UI ---
st.title("Food Recognition & Nutrition Estimator 🍽️")
st.write("Upload a food image. Get instant prediction and estimated nutrition.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)
    input_img = transform(img).unsqueeze(0)  # add batch dimension

    # Prediction
    with torch.no_grad():
        output = model(input_img)
        pred_idx = torch.argmax(output, 1).item()
        pred_class = FOOD_CLASSES[pred_idx]

    st.markdown(f"### Prediction: **{pred_class.replace('_',' ').title()}**")
    nutrition = lookup_nutrition(pred_class)
    st.markdown("#### Estimated Nutrition Per Serving:")
    st.table(pd.DataFrame([nutrition]))
else:
    st.write("Upload an image to begin.")