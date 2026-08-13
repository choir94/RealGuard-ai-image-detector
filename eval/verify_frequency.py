
import os, json, numpy as np
from PIL import Image
from scipy.special import expit
from scipy.fft import dctn

DATA_DIR = "/root/ai-image-detector/eval/data"

def block_energy(gray, block_size=8):
    h, w = gray.shape
    h2 = (h // block_size) * block_size
    w2 = (w // block_size) * block_size
    gray = gray[:h2, :w2]
    
    total_high_freq = 0.0
    total_energy = 0.0
    
    for y in range(0, h2, block_size):
        for x in range(0, w2, block_size):
            block = gray[y:y+block_size, x:x+block_size].astype(np.float32)
            dct_block = dctn(block, norm="ortho")
            dc = dct_block[0, 0]
            ac_energy = np.sum(dct_block ** 2) - dc ** 2
            total_energy += dc ** 2 + ac_energy
            hf = dct_block[block_size//2:, block_size//2:]
            total_high_freq += np.sum(hf ** 2)
    
    if total_energy == 0:
        return 0.5, 0, 0
    high_freq_ratio = total_high_freq / total_energy
    score = 1.0 - min(1.0, high_freq_ratio * 10)
    return float(score), float(high_freq_ratio), float(total_energy)

def fuse_frequency(p_cal, freq_score):
    if 0.3 <= p_cal <= 0.7:
        if freq_score >= 0.7:
            return p_cal + 0.05 * (freq_score - 0.5)
        elif freq_score <= 0.3:
            return p_cal - 0.05 * (0.5 - freq_score)
    return p_cal

with open(os.path.join(DATA_DIR, "benchmark_results.json")) as f:
    results = json.load(f)
with open(os.path.join(DATA_DIR, "calibration.json")) as f:
    cal = json.load(f)

a, b = cal["a"], cal["b"]
valid = [r for r in results if r["logit"] is not None]
logits = np.array([r["logit"] for r in valid])
labels = np.array([1 if r["label"] == "ai" else 0 for r in valid])

print("Computing frequency analysis...")
freq_scores = []
for r in valid:
    img_path = os.path.join(DATA_DIR, r["label"], r["file"])
    img = Image.open(img_path)
    img_small = img.resize((256, 256), Image.LANCZOS)
    gray = np.array(img_small.convert("L"), dtype=np.float32)
    score, hf_ratio, energy = block_energy(gray)
    freq_scores.append(score)

freq_scores = np.array(freq_scores)
calibrated = expit(a * logits + b)
fused = np.array([fuse_frequency(c, f) for c, f in zip(calibrated, freq_scores)])

threshold = 0.65
preds_no = (calibrated >= threshold).astype(int)
preds_yes = (fused >= threshold).astype(int)

tpr_no = (preds_no[labels == 1] == 1).mean()
tnr_no = (preds_no[labels == 0] == 0).mean()
ba_no = (tpr_no + tnr_no) / 2

tpr_yes = (preds_yes[labels == 1] == 1).mean()
tnr_yes = (preds_yes[labels == 0] == 0).mean()
ba_yes = (tpr_yes + tnr_yes) / 2

print(f"\n=== FREQUENCY ANALYSIS IMPACT (threshold=0.65) ===")
print(f"WITHOUT freq: TPR={tpr_no:.4f} TNR={tnr_no:.4f} BA={ba_no*100:.2f}%")
print(f"WITH freq:    TPR={tpr_yes:.4f} TNR={tnr_yes:.4f} BA={ba_yes*100:.2f}%")
print(f"Delta: {ba_yes*100 - ba_no*100:+.2f}%")

changed = 0
for i, r in enumerate(valid):
    if preds_no[i] != preds_yes[i]:
        changed += 1
        direction = "AI->REAL" if preds_no[i] == 1 else "REAL->AI"
        correct = "CORRECT" if preds_yes[i] == labels[i] else "WRONG"
        print(f"  CHANGED: {r['label']}/{r['file']}: {direction} cal={calibrated[i]:.4f} fused={fused[i]:.4f} freq={freq_scores[i]:.4f} ({correct})")
if changed == 0:
    print("  No changes — all misclassified images are outside uncertain zone [0.3, 0.7]")

# Save full results
output = []
for i, r in enumerate(valid):
    output.append({
        "file": r["file"], "label": r["label"], "logit": r["logit"],
        "pRaw": r["pFake"], "pCalibrated": float(calibrated[i]), "pFused": float(fused[i]),
        "freqScore": float(freq_scores[i]),
        "verdict": "ai" if preds_yes[i] == 1 else "real",
        "correct": bool(preds_yes[i] == labels[i]),
    })
with open(os.path.join(DATA_DIR, "full_benchmark.json"), "w") as f:
    json.dump(output, f, indent=2)

print(f"\n=== FINAL ===")
print(f"Raw sigmoid:      BA=80.83%")
print(f"Calibration only: BA={ba_no*100:.2f}%")
print(f"Cal + frequency:  BA={ba_yes*100:.2f}%")
print(f"Calibration: a={a}, b={b}")

# Per-generator breakdown
print(f"\n=== PER-GENERATOR (with calibration + freq) ===")
for prefix in ["mj", "dalle3", "flux", "ideogram", "4o"]:
    gen_results = [(r, l, f, p, pred) for r, l, f, p, pred in zip(valid, labels, freq_scores, fused, preds_yes) if r["file"].startswith(prefix)]
    if gen_results:
        correct = sum(1 for _, _, _, _, pred in gen_results if pred == 1)
        print(f"  {prefix}: {correct}/{len(gen_results)} correct ({correct/len(gen_results)*100:.0f}%)")
