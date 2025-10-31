import cv2
import numpy as np
import joblib
import ollama  # Local LLM
from skimage.measure import shannon_entropy
import tkinter as tk
from tkinter import filedialog, Label, Button, scrolledtext
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

    return {
        "Sharpness (Laplacian)": laplacian_var,
        "Entropy": entropy,
        "Contrast": contrast,
        "Brightness": brightness,
        "Edge Density": edge_density
    }

# === LLM Improvement Suggestions ===
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
        # On Mac/Linux Ollama returns a list of dicts
        if isinstance(response, dict) and "message" in response:
            return response["message"]["content"]
        elif isinstance(response, list) and len(response) > 0:
            return response[0]["message"]["content"]
        else:
            return "⚠️ No response from Ollama."
    except Exception as e:
        return f"Error generating suggestions: {str(e)}\n\n(AI not found)"

def select_image():
    file_path = filedialog.askopenfilename(
        title="Choose an image", 
        filetypes=[("Image files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        image = cv2.imread(file_path)
        if image is not None:
            features = extract_features(image)
            mos_score = model.predict([list(features.values())])[0] * 20

            # Load with Pillow for Tkinter
            img = Image.open(file_path)
            img.thumbnail((300, 300))
            img_tk = ImageTk.PhotoImage(img)

            # Keep reference to avoid garbage collection on macOS
            image_label.config(image=img_tk)
            image_label.image = img_tk  

            # Show results
            feature_text = "\n".join([f"{name}: {value:.4f}" for name, value in features.items()])
            result_label.config(
                text=f"Predicted Quality Score: {mos_score:.2f}/100\n\nFeature Values:\n{feature_text}"
            )

            # Show suggestions
            suggestion_text.delete(1.0, tk.END)
            suggestion_text.insert(tk.INSERT, "Generating suggestions...\n")
            app.update_idletasks()  # refresh UI on mac
            suggestions = get_improvement_suggestions(features, mos_score)
            suggestion_text.delete(1.0, tk.END)
            suggestion_text.insert(tk.INSERT, f"Improvement Suggestions:\n\n{suggestions}")
        else:
            result_label.config(text="⚠️ Could not read the image.")
            suggestion_text.delete(1.0, tk.END)
    else:
        result_label.config(text="⚠️ No file selected.")
        suggestion_text.delete(1.0, tk.END)

# === Build UI ===
app = tk.Tk()
app.title("Image Quality Analyzer (Local AI)")
app.geometry("700x800")
app.configure(bg="white")  # Force light background to make text visible

# Image selection button
Button(app, text="Select Image", command=select_image, font=("Arial", 14)).pack(pady=20)

# Image display
image_label = Label(app, bg="white")   # white background so image is visible
image_label.pack(pady=10, fill="both", expand=True)

# Results display
result_label = Label(app, text="", font=("Arial", 12), justify=tk.LEFT, bg="white", anchor="w")
result_label.pack(pady=10, fill="x")

# Suggestions display
suggestion_label = Label(app, text="AI Improvement Suggestions:", font=("Arial", 12, "bold"), bg="white", anchor="w")
suggestion_label.pack(pady=5, fill="x")

suggestion_text = scrolledtext.ScrolledText(
    app, wrap=tk.WORD, width=80, height=15, font=("Arial", 11), bg="white"
)
suggestion_text.pack(pady=10, fill="both", expand=True)

app.mainloop()
