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
import torch
from diffusers import StableDiffusionPipeline

# Load your model for predicted MOS
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

# Initialize Stable Diffusion model once
print("Loading Stable Diffusion model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5"
).to(device)

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

def get_improvement_prompt(features, score):
    feature_text = "\n".join([f"- {name}: {value:.2f}" for name, value in features.items()])
    prompt = f"""
You are an AI image editor.

Analyze these metrics and describe an improved version of the image in one short descriptive prompt usable by a Stable Diffusion model.

Quality Score: {score:.2f}/100
Features:
{feature_text}

Describe desired enhancements: brightness, clarity, color balance, detail, tone, exposure, and contrast.
Output only the final prompt.
"""
    try:
        response = ollama.chat(
            model='mistral',
            messages=[
                {"role": "system", "content": "You generate prompts for stable diffusion image enhancement."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["message"].content
    except Exception as e:
        return f"Error generating prompt: {e}"

def generate_image_with_diffusers(prompt):
    """Use local Stable Diffusion to generate the image through Hugging Face diffusers."""
    try:
        with torch.inference_mode():
            image = pipe(prompt).images[0]
            return image
    except Exception as e:
        print("Stable Diffusion error:", e)
        return None

def fetch_and_show_results(features, mos_score):
    suggestion_text.delete(1.0, tk.END)
    suggestion_text.insert(tk.INSERT, "Generating AI enhancement prompt...\n")

    ai_prompt = get_improvement_prompt(features, mos_score)
    suggestion_text.insert(tk.INSERT, "\nGenerated Prompt:\n\n")
    suggestion_text.insert(tk.INSERT, ai_prompt)
    suggestion_text.insert(tk.INSERT, "\n\nGenerating Enhanced Image using Stable Diffusion...\n")

    enhanced_image = generate_image_with_diffusers(ai_prompt)
    if enhanced_image:
        enhanced_image.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(enhanced_image)
        generated_image_label.config(image=img_tk)
        generated_image_label.image = img_tk
        suggestion_text.insert(tk.INSERT, "\n✅ Image generation completed successfully.")
    else:
        suggestion_text.insert(tk.INSERT, "\n❌ Failed to generate the enhanced image.")

def select_image():
    result_text.delete(1.0, tk.END)
    suggestion_text.delete(1.0, tk.END)
    generated_image_label.config(image='')

    if load_error:
        result_text.insert(tk.INSERT, f"⚠️ Model load failed: {load_error}\n")
        return

    file_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if not file_path:
        result_text.insert(tk.INSERT, "No file selected.\n")
        return

    image = cv2.imread(file_path)
    if image is None:
        result_text.insert(tk.INSERT, "Failed to load image.\n")
        return

    # Display original image
    img = Image.open(file_path)
    img.thumbnail((300, 300))
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    features = extract_features(image)
    result_text.insert(tk.INSERT, f"\nExtracted Features:\n")
    for k, v in features.items():
        result_text.insert(tk.INSERT, f"{k}: {v:.4f}\n")

    if model is None:
        result_text.insert(tk.INSERT, "\nModel not loaded; skipping MOS prediction.\n")
        return

    feature_vector = [features[feat] for feat in feature_order]
    mos_score = model.predict([feature_vector])[0] * 20
    result_text.insert(tk.INSERT, f"\nPredicted MOS Quality: {mos_score:.2f}/100\n")

    suggestion_text.insert(tk.INSERT, "Generating prompt and image...\n")
    threading.Thread(target=fetch_and_show_results, args=(features, mos_score), daemon=True).start()

# GUI setup
app = tk.Tk()
app.title("AI Image Quality Analyzer + Local Image Generator")
app.geometry("700x850")

button = tk.Button(app, text="Select Image & Generate Enhanced Image", command=select_image)
button.pack(pady=10)

image_label = tk.Label(app, bg="white")
image_label.pack(pady=10)

result_text = ScrolledText(app, width=70, height=15)
result_text.pack(pady=5)

suggestion_text = ScrolledText(app, width=70, height=10)
suggestion_text.pack(pady=5)

generated_image_label = tk.Label(app, bg="white")
generated_image_label.pack(pady=10)

app.mainloop()
