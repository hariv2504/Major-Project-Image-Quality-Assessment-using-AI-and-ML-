import cv2
import numpy as np
from skimage.measure import shannon_entropy
import joblib
import tkinter as tk
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import threading
import ollama

try:
    model = joblib.load("mos_predictor_model.pkl")
except Exception as e:
    model = None
    load_error = str(e)
else:
    load_error = None

feature_order = [
    "Sharpness (Laplacian)",
    "Entropy",
    "Contrast",
    "Brightness",
    "Edge Density"
]

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    entropy = shannon_entropy(gray)
    contrast = gray.std()
    brightness = np.mean(gray)
    edges = cv2.Canny(blur, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size

    return {
        "Sharpness (Laplacian)": laplacian_var,
        "Entropy": entropy,
        "Contrast": contrast,
        "Brightness": brightness,
        "Edge Density": edge_density
    }

def get_improvement_suggestions(features, score):
    feature_text = "\n".join([f"- {name}: {value:.2f}" for name, value in features.items()])
    prompt = f"""Analyze these image quality metrics and suggest specific technical improvements:

Current Quality Score: {score:.2f}/100
Image Features:
{feature_text}

Provide 3-5 concrete, actionable suggestions to improve this image's quality.
Focus on adjustments like brightness, contrast, sharpening, or noise reduction.
Format as bullet points with brief explanations.
"""
    try:
        response = ollama.chat(
            model='mistral',
            messages=[
                {"role": "system", "content": "You are a professional image editing assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["message"].content

    except Exception as e:
        return f"Error generating suggestions: {str(e)}\n\n(AI not found)"
    
def fetch_and_show_suggestions(features, mos_score):
    suggestions = get_improvement_suggestions(features, mos_score)
    suggestion_text.delete(1.0, tk.END)
    suggestion_text.insert(tk.INSERT, suggestions)

def select_image():
    result_text.delete(1.0, tk.END)
    suggestion_text.delete(1.0, tk.END)

    if load_error:
        result_text.insert(tk.INSERT, f"⚠️ Model load failed: {load_error}\n")
        return

    file_path = filedialog.askopenfilename(
        title="Choose an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        result_text.insert(tk.INSERT, "No file selected.\n")
        return

    result_text.insert(tk.INSERT, f"Selected file:\n{file_path}\n")

    image = cv2.imread(file_path)
    if image is None:
        result_text.insert(tk.INSERT, "Failed to load image via OpenCV.\n")
        return

    result_text.insert(tk.INSERT, f"Image shape: {image.shape}, dtype: {image.dtype}\n")

    img = Image.open(file_path)
    img.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk  # Keep reference

    features = extract_features(image)
    result_text.insert(tk.INSERT, "\nExtracted Features:\n")
    for k, v in features.items():
        result_text.insert(tk.INSERT, f"{k}: {v:.4f}\n")

    feature_vector = [features[feat] for feat in feature_order]
    mos_score = model.predict([feature_vector])[0] * 20
    result_text.insert(tk.INSERT, f"\nPredicted Quality Score (MOS): {mos_score:.2f}/100\n")

    suggestion_text.insert(tk.INSERT, "Generating suggestions from AI...\n")
    app.update_idletasks()  # Refresh UI before blocking call
    threading.Thread(target=fetch_and_show_suggestions, args=(features, mos_score), daemon=True).start()

app = tk.Tk()
app.title("Image Quality Analyzer with AI Suggestions")
app.geometry("600x750")
button = tk.Button(app, text="Select Image", command=select_image)
button.pack(pady=10)
image_label = tk.Label(app, bg="white")
image_label.pack(pady=10, fill="both", expand=False)
result_text = ScrolledText(app, width=60, height=15)
result_text.pack(pady=5, fill="both", expand=False)
suggestion_text = ScrolledText(app, width=60, height=8)
suggestion_text.pack(pady=5, fill="both", expand=True)

app.mainloop()
