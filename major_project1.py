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
import re
import traceback

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

current_image = None
current_image_path = None
last_modified_image = None

def extract_features(image):
    try:
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
    except Exception as e:
        print(f"Error in extract_features: {e}")
        return None

def get_improvement_suggestions(features, score):
    feature_text = "\n".join([f"- {name}: {value:.2f}" for name, value in features.items()])
    prompt = f"""Analyze these image quality metrics and suggest SPECIFIC improvements with EXACT numeric values:

Current Quality Score: {score:.2f}/100
Image Features:
{feature_text}

Provide 3-5 suggestions with NUMERIC values. Use EXACTLY this format:
1. Brightness: increase by 15%
2. Contrast: increase by 20%
3. Sharpness: increase with strength 1.5
4. Saturation: increase by 10%

Be specific with numbers. Always include the percentage (%) or strength value.
"""
    try:
        response = ollama.chat(
            model='mistral',
            messages=[
                {"role": "system", "content": "You are a professional image editor. Always provide specific numeric values."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["message"]["content"]
        
    except Exception as e:
        return "ollama error"

def parse_ai_suggestions(suggestions_text):
    params = {'brightness': 0, 'contrast': 0, 'sharpness': 0, 'saturation': 0, 'denoise': 0}
    try:
        brightness_match = re.search(r'brightness.*?(increase|decrease).*?by.*?(\d+)', suggestions_text, re.IGNORECASE)
        if brightness_match:
            direction = brightness_match.group(1).lower()
            value = int(brightness_match.group(2))
            params['brightness'] = value if direction == 'increase' else -value
        contrast_match = re.search(r'contrast.*?(increase|decrease).*?by.*?(\d+)', suggestions_text, re.IGNORECASE)
        if contrast_match:
            direction = contrast_match.group(1).lower()
            value = int(contrast_match.group(2))
            params['contrast'] = value if direction == 'increase' else -value
        sharpness_match = re.search(r'sharp.*?(?:strength|by).*?(\d+\.?\d*)', suggestions_text, re.IGNORECASE)
        if sharpness_match:
            params['sharpness'] = float(sharpness_match.group(1))
        saturation_match = re.search(r'saturation.*?(increase|decrease).*?by.*?(\d+)', suggestions_text, re.IGNORECASE)
        if saturation_match:
            direction = saturation_match.group(1).lower()
            value = int(saturation_match.group(2))
            params['saturation'] = value if direction == 'increase' else -value
    except Exception as e:
        print(f"Error parsing suggestions: {e}")
    return params

def adjust_brightness(image, value):
    if value == 0:
        return image
    value = max(-50, min(50, value))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] *= (1 + value / 100.0)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def adjust_contrast(image, value):
    if value == 0:
        return image
    value = max(-50, min(50, value))
    factor = 1 + value / 100.0
    mean = np.mean(image)
    adjusted = (image.astype(np.float32) - mean) * factor + mean
    return np.clip(adjusted, 0, 255).astype(np.uint8)

def adjust_sharpness(image, strength):
    if strength == 0:
        return image
    strength = max(0.1, min(3.0, strength))
    gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
    sharpened = cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def adjust_saturation(image, value):
    if value == 0:
        return image
    value = max(-50, min(50, value))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= (1 + value / 100.0)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_ai_edits(image, params):
    edited = image.copy()
    if params['brightness'] != 0:
        edited = adjust_brightness(edited, params['brightness'])
    if params['contrast'] != 0:
        edited = adjust_contrast(edited, params['contrast'])
    if params['sharpness'] > 0:
        edited = adjust_sharpness(edited, params['sharpness'])
    if params['saturation'] != 0:
        edited = adjust_saturation(edited, params['saturation'])
    return edited

def save_modified_image():
    global last_modified_image
    if last_modified_image is None:
        suggestion_text.insert(tk.END, "\n No modified image to save.")
        return
    output_path = filedialog.asksaveasfilename(defaultextension=".jpg",
        filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
    if output_path:
        cv2.imwrite(output_path, last_modified_image)
        suggestion_text.insert(tk.END, f"\n💾 Saved to: {output_path}")


def apply_modifications():
    global last_modified_image, current_image
    if current_image is None:
        suggestion_text.insert(tk.END, "\n No image loaded.")
        return
    suggestions = suggestion_text.get(1.0, tk.END)
    params = parse_ai_suggestions(suggestions)
    modified_image = apply_ai_edits(current_image, params)
    last_modified_image = modified_image.copy()
    display_image(modified_image, modified_image_label)
    suggestion_text.insert(tk.END, "\n\n AI Changes Applied")


def fetch_and_show_suggestions(features, mos_score):
    suggestions = get_improvement_suggestions(features, mos_score)
    suggestion_text.delete(1.0, tk.END)
    suggestion_text.insert(tk.INSERT, suggestions)
    suggestion_text.insert(tk.END, "\n\n" + "="*50)
    suggestion_text.insert(tk.END, "\n Click 'Apply AI Edits' to see results")


def fetch_suggestions_manual():
    """Manually fetch AI suggestions without auto-running on image load"""
    global current_image
    if current_image is None:
        suggestion_text.delete(1.0, tk.END)
        suggestion_text.insert(tk.END, " Please select an image first")
        return
    
    suggestion_text.delete(1.0, tk.END)
    suggestion_text.insert(tk.END, " Generating AI suggestions...\n")
    app.update_idletasks()
    
    features = extract_features(current_image)
    if features is None:
        suggestion_text.delete(1.0, tk.END)
        suggestion_text.insert(tk.END, "Feature extraction failed")
        return
    
    mos_score = model.predict([[features[f] for f in feature_order]])[0] * 20 if model else 50
    fetch_and_show_suggestions(features, mos_score)


def display_image(image, label):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    pil_img.thumbnail((800, 600))
    tk_img = ImageTk.PhotoImage(pil_img)
    label.config(image=tk_img)
    label.image = tk_img


def select_image():
    global current_image, current_image_path
    result_text.delete(1.0, tk.END)
    suggestion_text.delete(1.0, tk.END)
    modified_image_label.config(image='')
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return
    image = cv2.imread(file_path)
    if image is None:
        result_text.insert(tk.END, " Failed to load image\n")
        return
    current_image = image.copy()
    current_image_path = file_path
    display_image(image, image_label)
    features = extract_features(image)
    if features is None:
        result_text.insert(tk.END, "Feature extraction failed\n")
        return
    mos_score = model.predict([[features[f] for f in feature_order]])[0] * 20 if model else 50
    result_text.insert(tk.END, f" Features:\n")
    for k, v in features.items():
        result_text.insert(tk.END, f"  • {k}: {v:.2f}\n")
    result_text.insert(tk.END, f"\n MOS Score: {mos_score:.1f}/100\n")
    suggestion_text.insert(tk.END, " Click 'AI Enhanced Image' to generate suggestions\n")

app = tk.Tk()
app.title("Image Quality Assesment ")
app.geometry("1500x950")
app.configure(bg="#000000")

button_frame = tk.Frame(app, bg="#000000")
button_frame.pack(side=tk.TOP, fill="x", pady=10)
tk.Button(button_frame, text=" Select Image", command=select_image,
          font=("Arial", 12, "bold"), bg="#1B5E20", fg="black",
          activebackground="#2E7D32", padx=20, pady=8).pack(side=tk.LEFT, padx=8)
tk.Button(button_frame, text=" Get AI Suggestions", command=lambda: threading.Thread(target=fetch_suggestions_manual, daemon=True).start(),
          font=("Arial", 12, "bold"), bg="#7B1FA2", fg="black",
          activebackground="#9C27B0", padx=20, pady=8).pack(side=tk.LEFT, padx=8)
tk.Button(button_frame, text="Apply Changes", command=apply_modifications,
          font=("Arial", 12, "bold"), bg="#0D47A1", fg="black",
          activebackground="#1565C0", padx=20, pady=8).pack(side=tk.LEFT, padx=8)
tk.Button(button_frame, text=" Save Modified", command=save_modified_image,
          font=("Arial", 12, "bold"), bg="#E65100", fg="black",
          activebackground="#EF6C00", padx=20, pady=8).pack(side=tk.LEFT, padx=8)

canvas = tk.Canvas(app, bg="#000000", highlightthickness=0)
scrollbar_y = tk.Scrollbar(app, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg="#000000")


scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar_y.set)


canvas.pack(side=tk.LEFT, fill="both", expand=True)
scrollbar_y.pack(side=tk.RIGHT, fill="y")

image_frame = tk.Frame(scrollable_frame, bg="#000000")
image_frame.pack(fill="x", pady=15)
original_col = tk.Frame(image_frame, bg="#000000")
original_col.pack(side=tk.LEFT, fill="both", expand=True, padx=20)
tk.Label(original_col, text="Selected Image", font=("Arial", 13, "bold"), fg="#00E676", bg="#000000").pack(pady=5)
image_label = tk.Label(original_col, bg="#111111", relief="ridge", bd=3, width=600, height=400)
image_label.pack(pady=5)
modified_col = tk.Frame(image_frame, bg="#000000")
modified_col.pack(side=tk.LEFT, fill="both", expand=True, padx=20)
tk.Label(modified_col, text="Modified Image", font=("Arial", 13, "bold"), fg="#00B0FF", bg="#000000").pack(pady=5)
modified_image_label = tk.Label(modified_col, bg="#111111", relief="ridge", bd=3, width=600, height=400)
modified_image_label.pack(pady=5)

info_frame = tk.Frame(scrollable_frame, bg="#111111", relief="groove", bd=3)
info_frame.pack(fill="x", pady=15, padx=20)
tk.Label(info_frame, text="Features Extracted ", font=("Arial", 12, "bold"), bg="#111111", fg="#FFFFFF").pack(anchor="w", pady=5)
result_text = ScrolledText(info_frame, width=140, height=12, font=("Courier", 10), wrap="word", bg="#000000", fg="#00E5FF", insertbackground="white", relief="flat")
result_text.pack(pady=5, fill="both", expand=True)
tk.Label(info_frame, text="AI Suggestions", font=("Arial", 12, "bold"), bg="#111111", fg="#FFFFFF").pack(anchor="w", pady=5)
suggestion_text = ScrolledText(info_frame, width=140, height=12, font=("Courier", 10), wrap="word", bg="#000000", fg="#69F0AE", insertbackground="white", relief="flat")
suggestion_text.pack(pady=5, fill="both", expand=True)


app.mainloop()
