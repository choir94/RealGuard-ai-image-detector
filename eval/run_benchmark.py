
import os, json, sys, time
import numpy as np
from PIL import Image
import onnxruntime as ort

MODEL_PATH = "/tmp/model.onnx"
DATA_DIR = "/root/ai-image-detector/eval/data"
INPUT_SIZE = 384
RESIZE = 440
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def bilinear_coefficients(input_size, output_size):
    """Pillow-matching bilinear resize coefficients."""
    scale = input_size / output_size
    filter_scale = max(1.0, scale)
    support = filter_scale
    coeffs = []
    for i in range(output_size):
        center = (i + 0.5) * scale
        start = max(0, int(np.floor(center - support + 0.5)))
        end = min(input_size, int(np.floor(center + support + 0.5)))
        weights = []
        total = 0.0
        for src in range(start, end):
            distance = abs((src + 0.5 - center) / filter_scale)
            weight = max(0.0, 1.0 - distance)
            weights.append(weight)
            total += weight
        coeffs.append((start, [w / total for w in weights]))
    return coeffs

def bilinear_resize_rgb(img_arr, target_w, target_h):
    """Bilinear resize matching Pillow. img_arr: HxWx3 float32."""
    h, w = img_arr.shape[:2]
    x_coeffs = bilinear_coefficients(w, target_w)
    y_coeffs = bilinear_coefficients(h, target_h)
    
    # Pass 1: Horizontal resize
    intermediate = np.zeros((h, target_w, 3), dtype=np.float32)
    for x, (start, weights) in enumerate(x_coeffs):
        for c in range(3):
            intermediate[:, x, c] = sum(img_arr[:, start + i, c] * w for i, w in enumerate(weights))
    
    # Pass 2: Vertical resize
    output = np.zeros((target_h, target_w, 3), dtype=np.float32)
    for y, (start, weights) in enumerate(y_coeffs):
        for c in range(3):
            output[y, :, c] = sum(intermediate[start + i, :, c] * w for i, w in enumerate(weights))
    
    return output

def preprocess(img):
    """Full preprocessing: resize → center crop → normalize → CHW tensor."""
    w, h = img.size
    img_arr = np.array(img, dtype=np.float32)  # HxWx3
    
    # Compute resize dimensions
    scale = RESIZE / min(w, h)
    rw = max(INPUT_SIZE, round(w * scale))
    rh = max(INPUT_SIZE, round(h * scale))
    
    # Custom bilinear resize
    resized = bilinear_resize_rgb(img_arr, rw, rh)
    
    # Center crop
    cx = (rw - INPUT_SIZE) // 2
    cy = (rh - INPUT_SIZE) // 2
    cropped = resized[cy:cy + INPUT_SIZE, cx:cx + INPUT_SIZE, :]
    
    # Normalize and convert to CHW
    cropped = cropped / 255.0
    cropped = (cropped - MEAN) / STD
    chw = np.transpose(cropped, (2, 0, 1))  # HWC → CHW
    return np.expand_dims(chw, 0).astype(np.float32)  # 1x3xHxW

def main():
    # Load model
    print("Loading model...")
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT/resolve/main/onnx/model.onnx",
            MODEL_PATH
        )
    
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Session ready. Input: {input_name}, Output: {output_name}")
    
    # Process all images
    results = []
    
    for label in ["real", "ai"]:
        dir_path = os.path.join(DATA_DIR, label)
        if not os.path.exists(dir_path):
            continue
        files = sorted([f for f in os.listdir(dir_path) if f.endswith(".jpg") and os.path.getsize(os.path.join(dir_path, f)) > 5000])
        
        for fname in files:
            fpath = os.path.join(dir_path, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                tensor = preprocess(img)
                outputs = session.run([output_name], {input_name: tensor})
                logit = float(outputs[0][0][0])
                p_fake = 1.0 / (1.0 + np.exp(-logit))
                
                results.append({"file": fname, "label": label, "logit": logit, "pFake": p_fake})
                print(f"  {label}/{fname}: logit={logit:.4f} p={p_fake:.4f}")
            except Exception as e:
                print(f"  {label}/{fname}: ERROR {e}")
                results.append({"file": fname, "label": label, "logit": None, "pFake": None, "error": str(e)})
    
    # Save results
    out_path = os.path.join(DATA_DIR, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{len(results)} results saved to {out_path}")
    
    # Summary at threshold 0.65
    valid = [r for r in results if r["logit"] is not None]
    real = [r for r in valid if r["label"] == "real"]
    ai = [r for r in valid if r["label"] == "ai"]
    
    threshold = 0.65
    real_correct = sum(1 for r in real if r["pFake"] < threshold)
    ai_correct = sum(1 for r in ai if r["pFake"] >= threshold)
    tpr = ai_correct / len(ai) if ai else 0  # AI recall
    tnr = real_correct / len(real) if real else 0  # Real recall
    balanced_acc = (tpr + tnr) / 2
    
    print(f"\n=== SUMMARY (threshold={threshold}) ===")
    print(f"Real: {real_correct}/{len(real)} correct (TNR={tnr:.4f})")
    print(f"AI: {ai_correct}/{len(ai)} correct (TPR={tpr:.4f})")
    print(f"Balanced accuracy: {balanced_acc * 100:.2f}%")
    
    # Also compute at multiple thresholds
    print(f"\n=== THRESHOLD SWEEP ===")
    for t in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]:
        rc = sum(1 for r in real if r["pFake"] < t)
        ac = sum(1 for r in ai if r["pFake"] >= t)
        tr = ac / len(ai) if ai else 0
        fr = rc / len(real) if real else 0
        ba = (tr + fr) / 2
        print(f"  t={t:.2f}: TPR={tr:.4f} TNR={fr:.4f} BA={ba*100:.2f}%")

main()
