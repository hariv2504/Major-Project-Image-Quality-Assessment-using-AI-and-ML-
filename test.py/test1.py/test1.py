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


# Global variables
current_image = None
current_image_path = None
last_modified_image = None


def extract_features(image):
    """Extract image quality features"""
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
    """Get AI suggestions with specific parameters"""
    feature_text = "\\n".join([f"- {name}: {value:.2f}" for name, value in features.items()])
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
        error_msg = f"⚠️ Ollama not available. Using metric-based suggestions:\\n\\n"
        
        suggestions = []
        if features["Brightness"] < 100:
            suggestions.append("1. Brightness: increase by 15%")
        elif features["Brightness"] > 150:
            suggestions.append("1. Brightness: decrease by 10%")
        else:
            suggestions.append("1. Brightness: maintain current level")
            
        if features["Contrast"] < 30:
            suggestions.append("2. Contrast: increase by 25%")
        else:
            suggestions.append("2. Contrast: increase by 10%")
            
        if features["Sharpness (Laplacian)"] < 100:
            suggestions.append("3. Sharpness: increase with strength 1.8")
        else:
            suggestions.append("3. Sharpness: increase with strength 1.2")
            
        suggestions.append("4. Saturation: increase by 15%")
        
        return error_msg + "\\n".join(suggestions)


def parse_ai_suggestions(suggestions_text):
    """Parse AI suggestions to extract editing parameters"""
    params = {
        'brightness': 0,
        'contrast': 0,
        'sharpness': 0,
        'saturation': 0,
        'denoise': 0
    }
    
    try:
        brightness_match = re.search(r'brightness.*?(increase|decrease).*?by.*?(\\d+)', suggestions_text, re.IGNORECASE)
        if brightness_match:
            direction = brightness_match.group(1).lower()
            value = int(brightness_match.group(2))
            params['brightness'] = value if direction == 'increase' else -value
        
        contrast_match = re.search(r'contrast.*?(increase|decrease).*?by.*?(\\d+)', suggestions_text, re.IGNORECASE)
        if contrast_match:
            direction = contrast_match.group(1).lower()
            value = int(contrast_match.group(2))
            params['contrast'] = value if direction == 'increase' else -value
        
        sharpness_match = re.search(r'sharp.*?(?:strength|by).*?(\\d+\\.?\\d*)', suggestions_text, re.IGNORECASE)
        if sharpness_match:
            params['sharpness'] = float(sharpness_match.group(1))
        
        saturation_match = re.search(r'saturation.*?(increase|decrease).*?by.*?(\\d+)', suggestions_text, re.IGNORECASE)
        if saturation_match:
            direction = saturation_match.group(1).lower()
            value = int(saturation_match.group(2))
            params['saturation'] = value if direction == 'increase' else -value
        
        if re.search(r'denoise.*?yes|noise.*?reduc', suggestions_text, re.IGNORECASE):
            denoise_match = re.search(r'denoise.*?(?:strength|by).*?(\\d+)', suggestions_text, re.IGNORECASE)
            params['denoise'] = int(denoise_match.group(1)) if denoise_match else 5
            
    except Exception as e:
        print(f"Error parsing suggestions: {e}")
    
    return params


def adjust_brightness(image, value):
    """Adjust brightness by percentage (-50 to +50)"""
    try:
        if value == 0:
            return image
        value = max(-50, min(50, value))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * (1 + value / 100.0)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    except Exception as e:
        print(f"Error adjusting brightness: {e}")
        return image


def adjust_contrast(image, value):
    """Adjust contrast by percentage (-50 to +50)"""
    try:
        if value == 0:
            return image
        value = max(-50, min(50, value))
        factor = 1 + value / 100.0
        mean = np.mean(image)
        adjusted = (image.astype(np.float32) - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"Error adjusting contrast: {e}")
        return image


def adjust_sharpness(image, strength):
    """Apply sharpening filter"""
    try:
        if strength == 0:
            return image
        strength = max(0.1, min(3.0, strength))
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.0 + strength, gaussian, -strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    except Exception as e:
        print(f"Error adjusting sharpness: {e}")
        return image


def adjust_saturation(image, value):
    """Adjust color saturation by percentage (-50 to +50)"""
    try:
        if value == 0:
            return image
        value = max(-50, min(50, value))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * (1 + value / 100.0)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    except Exception as e:
        print(f"Error adjusting saturation: {e}")
        return image


def apply_denoise(image, strength):
    """Apply denoising filter"""
    try:
        if strength == 0:
            return image
        strength = max(1, min(20, int(strength)))
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    except Exception as e:
        print(f"Error applying denoise: {e}")
        return image


def apply_ai_edits(image, params):
    """Apply all editing parameters to the image"""
    try:
        edited = image.copy()
        
        if params['brightness'] != 0:
            edited = adjust_brightness(edited, params['brightness'])
        if params['contrast'] != 0:
            edited = adjust_contrast(edited, params['contrast'])
        if params['sharpness'] > 0:
            edited = adjust_sharpness(edited, params['sharpness'])
        if params['saturation'] != 0:
            edited = adjust_saturation(edited, params['saturation'])
        if params['denoise'] > 0:
            edited = apply_denoise(edited, params['denoise'])
        
        return edited
    except Exception as e:
        print(f"Error applying AI edits: {e}")
        traceback.print_exc()
        return image


def save_modified_image():
    """Save the currently displayed modified image"""
    try:
        if last_modified_image is None:
            suggestion_text.insert(tk.END, "\\n\\n❌ No modified image to save.")
            return
        
        output_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")],
            initialfile=current_image_path.rsplit('/', 1)[-1].rsplit('.', 1)[0] + '_modified.jpg'
        )
        
        if output_path:
            cv2.imwrite(output_path, last_modified_image)
            suggestion_text.insert(tk.END, f"\\n\\n💾 Saved to: {output_path}")
    except Exception as e:
        suggestion_text.insert(tk.END, f"\\n\\n❌ Error: {str(e)}")


def apply_modifications():
    """Apply AI suggestions and display in GUI"""
    global current_image, last_modified_image
    
    try:
        if current_image is None:
            suggestion_text.insert(tk.END, "\\n\\n❌ No image loaded.")
            return
        
        suggestions = suggestion_text.get(1.0, tk.END)
        if len(suggestions.strip()) < 10:
            suggestion_text.insert(tk.END, "\\n\\n❌ No valid suggestions.")
            return
        
        suggestion_text.insert(tk.END, "\\n\\n🔄 Parsing suggestions...")
        app.update_idletasks()
        
        params = parse_ai_suggestions(suggestions)
        suggestion_text.insert(tk.END, f"\\n📊 Parameters: {params}")
        app.update_idletasks()
        
        if all(v == 0 for v in params.values()):
            suggestion_text.insert(tk.END, "\\n⚠️ Using defaults...")
            params = {'brightness': 10, 'contrast': 15, 'sharpness': 1.2, 'saturation': 10, 'denoise': 0}
        
        suggestion_text.insert(tk.END, "\\n🎨 Applying...")
        app.update_idletasks()
        
        modified_image = apply_ai_edits(current_image, params)
        last_modified_image = modified_image.copy()
        
        # Display at LARGER size - 600x600 instead of 400x400
        modified_rgb = cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB)
        modified_pil = Image.fromarray(modified_rgb)
        modified_pil.thumbnail((600, 600), Image.Resampling.LANCZOS)  # High quality resize
        img_tk = ImageTk.PhotoImage(modified_pil)
        
        modified_image_label.config(image=img_tk)
        modified_image_label.image = img_tk
        
        suggestion_text.insert(tk.END, "\\n\\n✅ Done! Compare images above.")
        suggestion_text.insert(tk.END, "\\n💡 Click 'Save' if you like the result.")
        
        current_image = modified_image
        
    except Exception as e:
        suggestion_text.insert(tk.END, f"\\n\\n❌ Error: {str(e)}")
        traceback.print_exc()


def fetch_and_show_suggestions(features, mos_score):
    """Fetch AI suggestions in background"""
    try:
        suggestions = get_improvement_suggestions(features, mos_score)
        suggestion_text.delete(1.0, tk.END)
        suggestion_text.insert(tk.INSERT, suggestions)
        suggestion_text.insert(tk.END, "\\n\\n" + "="*50)
        suggestion_text.insert(tk.END, "\\n👆 Click 'Apply AI Edits' to see results")
    except Exception as e:
        suggestion_text.delete(1.0, tk.END)
        suggestion_text.insert(tk.INSERT, f"Error: {str(e)}\\n")
        traceback.print_exc()


def select_image():
    """Handle image selection"""
    global current_image, current_image_path, last_modified_image
    
    try:
        result_text.delete(1.0, tk.END)
        suggestion_text.delete(1.0, tk.END)
        modified_image_label.config(image='')
        last_modified_image = None

        if load_error:
            result_text.insert(tk.INSERT, f"⚠️ Model issue: {load_error}\\n")

        file_path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            return

        current_image_path = file_path
        result_text.insert(tk.INSERT, f"📁 {file_path.split('/')[-1]}\\n")

        image = cv2.imread(file_path)
        if image is None:
            result_text.insert(tk.INSERT, "❌ Failed to load\\n")
            return

        current_image = image.copy()
        result_text.insert(tk.INSERT, f"✅ {image.shape[1]}x{image.shape[0]}px\\n")

        # Display at LARGER size - 600x600
        img = Image.open(file_path)
        img.thumbnail((600, 600), Image.Resampling.LANCZOS)  # High quality resize
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk

        features = extract_features(image)
        if features is None:
            result_text.insert(tk.INSERT, "❌ Feature extraction failed\\n")
            return
            
        result_text.insert(tk.INSERT, "\\n📊 Features:\\n")
        for k, v in features.items():
            result_text.insert(tk.INSERT, f"  • {k}: {v:.2f}\\n")

        if model is not None:
            feature_vector = [features[feat] for feat in feature_order]
            mos_score = model.predict([feature_vector])[0] * 20
            result_text.insert(tk.INSERT, f"\\n🎯 Score: {mos_score:.1f}/100\\n")
        else:
            mos_score = 50
            result_text.insert(tk.INSERT, f"\\n⚠️ Default: {mos_score}/100\\n")

        suggestion_text.insert(tk.INSERT, "🤖 Generating suggestions...\\n")
        app.update_idletasks()
        threading.Thread(target=fetch_and_show_suggestions, args=(features, mos_score), daemon=True).start()
        
    except Exception as e:
        result_text.insert(tk.END, f"\\n❌ {str(e)}\\n")
        traceback.print_exc()


# GUI Setup with LARGER window and images
app = tk.Tk()
app.title("Image Quality Analyzer with AI Editing")
app.geometry("1400x900")  # Much wider window

# Top buttons
button_frame = tk.Frame(app)
button_frame.pack(pady=15)

tk.Button(button_frame, text="📁 Select Image", command=select_image, 
         font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", 
         padx=20, pady=8).pack(side=tk.LEFT, padx=5)

tk.Button(button_frame, text="✨ Apply AI Edits", command=apply_modifications,
         font=("Arial", 12, "bold"), bg="#2196F3", fg="white", 
         padx=20, pady=8).pack(side=tk.LEFT, padx=5)

tk.Button(button_frame, text="💾 Save Modified", command=save_modified_image,
         font=("Arial", 12, "bold"), bg="#FF9800", fg="white", 
         padx=20, pady=8).pack(side=tk.LEFT, padx=5)

# LARGER image display area
image_frame = tk.Frame(app)
image_frame.pack(pady=15)

# Original image - BIGGER
original_col = tk.Frame(image_frame)
original_col.pack(side=tk.LEFT, padx=20)
tk.Label(original_col, text="📷 ORIGINAL IMAGE", 
        font=("Arial", 12, "bold"), fg="#2E7D32").pack()
image_label = tk.Label(original_col, bg="#E8F5E9", relief="ridge", bd=3)
image_label.config(width=600, height=600)  # Fixed large size
image_label.pack(pady=5)

# Modified image - BIGGER
modified_col = tk.Frame(image_frame)
modified_col.pack(side=tk.LEFT, padx=20)
tk.Label(modified_col, text="✨ MODIFIED IMAGE", 
        font=("Arial", 12, "bold"), fg="#1565C0").pack()
modified_image_label = tk.Label(modified_col, bg="#E3F2FD", relief="ridge", bd=3)
modified_image_label.config(width=600, height=600)  # Fixed large size
modified_image_label.pack(pady=5)

# Compact info sections below images
info_frame = tk.Frame(app)
info_frame.pack(pady=10, fill="both", expand=True)

# Analysis results - more compact
tk.Label(info_frame, text="📊 Analysis", font=("Arial", 10, "bold")).pack()
result_text = ScrolledText(info_frame, width=140, height=6, font=("Courier", 9))
result_text.pack(pady=3)

# AI suggestions - more compact
tk.Label(info_frame, text="🤖 AI Suggestions", font=("Arial", 10, "bold")).pack()
suggestion_text = ScrolledText(info_frame, width=140, height=6, font=("Courier", 9))
suggestion_text.pack(pady=3)

app.mainloop()
