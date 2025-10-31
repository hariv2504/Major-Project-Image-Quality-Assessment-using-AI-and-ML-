import cv2
import numpy as np
import joblib
from skimage.measure import shannon_entropy
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

# === Load Trained Model ===
model = joblib.load("mos_predictor_model.pkl")

# === Feature Extraction Function ===
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    entropy = shannon_entropy(gray)
    contrast = gray.std()
    brightness = np.mean(gray)
    edges = cv2.Canny(blur, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    return [laplacian_var, entropy, contrast, brightness, edge_density]

# === Predict Function ===
def select_image():
    file_path = filedialog.askopenfilename(title="Choose an image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        image = cv2.imread(file_path)
        if image is not None:
            features = extract_features(image)
            mos_score = model.predict([features])[0]*20
            
            # Create feature descriptions
            feature_names = [
                "Blur:",
                "Shannon Entropy:",
                "Contrast:",
                "Brightness:",
                "Edge Density:"
            ]
            
            # Format feature values
            feature_text = "\n".join([f"{name} {value:.4f}" for name, value in zip(feature_names, features)])
            
            # Display image
            img = Image.open(file_path)
            img.thumbnail((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            image_label.config(image=img_tk)
            image_label.image = img_tk

            # Display MOS and features
            result_label.config(text=f"Predicted MOS Score: {mos_score:.2f}\n\nFeature Values:\n{feature_text}")
        else:
            result_label.config(text="❌ Could not read the image.")
    else:
        result_label.config(text="⚠️ No file selected.")

# === Build UI ===
app = tk.Tk()
app.title("Image MOS Predictor")
app.geometry("400x550")  # Increased height to accommodate feature values

Button(app, text="Select Image", command=select_image, font=("Arial", 14)).pack(pady=20)
image_label = Label(app)
image_label.pack(pady=10)
result_label = Label(app, text="", font=("Arial", 12), justify=tk.LEFT)
result_label.pack()

app.mainloop()