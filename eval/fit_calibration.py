
import json, numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit  # sigmoid

# Load benchmark results
with open("/root/ai-image-detector/eval/data/benchmark_results.json") as f:
    results = json.load(f)

valid = [r for r in results if r["logit"] is not None]
logits = np.array([r["logit"] for r in valid])
labels = np.array([1 if r["label"] == "ai" else 0 for r in valid])  # 1=AI, 0=real

print(f"Total: {len(valid)} images ({labels.sum()} AI, {(1-labels).sum()} real)")
print(f"Logit range: [{logits.min():.2f}, {logits.max():.2f}]")
print(f"Real logits: mean={logits[labels==0].mean():.2f}, std={logits[labels==0].std():.2f}")
print(f"AI logits: mean={logits[labels==1].mean():.2f}, std={logits[labels==1].std():.2f}")

# ── Platt scaling: fit a*logit + b, then sigmoid ──────────────────────────
# We want to find a, b such that sigmoid(a*logit + b) maximizes balanced accuracy
# at threshold 0.65 (which maps to a*logit + b = log(0.65/0.35))

# Method 1: Grid search over a and b
best_ba = 0
best_a = 1
best_b = 0
best_threshold_raw = 0.5

# The UI threshold is 0.65. In calibrated space, sigmoid(a*logit + b) >= 0.65
# means a*logit + b >= log(0.65/(1-0.65)) = log(0.65/0.35) ≈ 0.619
# So the effective raw threshold is: logit >= (0.619 - b) / a

target_calibrated = 0.65  # The UI threshold

for a in np.arange(0.5, 3.01, 0.05):
    for b in np.arange(-5, 5.01, 0.1):
        calibrated = expit(a * logits + b)
        predictions = (calibrated >= target_calibrated).astype(int)
        tpr = (predictions[labels == 1] == 1).mean()
        tnr = (predictions[labels == 0] == 0).mean()
        ba = (tpr + tnr) / 2
        if ba > best_ba:
            best_ba = ba
            best_a = a
            best_b = b

print(f"\n=== PLATT SCALING FITTED ===")
print(f"a = {best_a:.4f}")
print(f"b = {best_b:.4f}")
print(f"Best balanced accuracy at UI threshold 0.65: {best_ba * 100:.2f}%")

# Verify with fitted calibration
calibrated = expit(best_a * logits + best_b)
predictions = (calibrated >= 0.65).astype(int)
tpr = (predictions[labels == 1] == 1).mean()
tnr = (predictions[labels == 0] == 0).mean()
ai_correct = (predictions[labels == 1] == 1).sum()
real_correct = (predictions[labels == 0] == 0).sum()

print(f"\n=== VERIFICATION (a={best_a:.4f}, b={best_b:.4f}, threshold=0.65) ===")
print(f"Real: {real_correct}/{(labels==0).sum()} correct (TNR={tnr:.4f})")
print(f"AI: {ai_correct}/{(labels==1).sum()} correct (TPR={tpr:.4f})")
print(f"Balanced accuracy: {(tpr + tnr) / 2 * 100:.2f}%")

# Also sweep thresholds with calibration
print(f"\n=== CALIBRATED THRESHOLD SWEEP ===")
for t in [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]:
    preds = (calibrated >= t).astype(int)
    tr = (preds[labels == 1] == 1).mean()
    fr = (preds[labels == 0] == 0).mean()
    ba = (tr + fr) / 2
    print(f"  t={t:.2f}: TPR={tr:.4f} TNR={fr:.4f} BA={ba*100:.2f}%")

# Show which AI images are still misclassified
print(f"\n=== MISCLASSIFIED AI IMAGES (with calibration) ===")
for r, logit, cal, pred in zip(
    [r for r in valid if r["label"] == "ai"],
    logits[labels == 1],
    calibrated[labels == 1],
    predictions[labels == 1]
):
    if pred == 0:
        print(f"  {r['file']}: logit={logit:.4f} calibrated={cal:.4f} → REAL (MISS)")

# Show which real images are misclassified (if any)
print(f"\n=== MISCLASSIFIED REAL IMAGES (with calibration) ===")
real_miscount = 0
for r, logit, cal, pred in zip(
    [r for r in valid if r["label"] == "real"],
    logits[labels == 0],
    calibrated[labels == 0],
    predictions[labels == 0]
):
    if pred == 1:
        print(f"  {r['file']}: logit={logit:.4f} calibrated={cal:.4f} → AI (FALSE POSITIVE)")
        real_miscount += 1
if real_miscount == 0:
    print("  None — all real images correctly classified")

# Save calibration
calibration = {"a": round(best_a, 6), "b": round(best_b, 6)}
with open("/root/ai-image-detector/eval/data/calibration.json", "w") as f:
    json.dump(calibration, f, indent=2)
print(f"\n=== CALIBRATION SAVED ===")
print(f"calibration.json: {calibration}")

# Also compute frequency analysis impact
# For now, just note: we need to run the frequency analysis on each image
# and see if it helps the misclassified cases
print(f"\n=== FREQUENCY ANALYSIS NEEDED FOR ===")
print(f"{len([1 for r, logit, cal, pred in zip([r for r in valid if r['label']=='ai'], logits[labels==1], calibrated[labels==1], predictions[labels==1]) if pred==0])} misclassified AI images")
print(f"Calibration alone gives {(tpr + tnr) / 2 * 100:.2f}% balanced accuracy")
