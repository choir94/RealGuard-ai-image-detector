# Privacy Policy

The AI Image Detector extension is designed so that **detection happens
entirely on your device**. Your images are not uploaded to any server for
analysis.

---

## 1. All Detection Is Local

Every inference runs **inside the extension** using ONNX Runtime Web. The
neural-network model executes in your browser's WASM/WebGPU runtime. No
image pixels are ever transmitted to a remote endpoint for classification.

---

## 2. Image Fetching Is for Local Pixel Inference Only

When you trigger detection on an image, the extension fetches the image
bytes **from the original host that already served the page**. This fetch
happens so the extension can read pixel data locally and feed it into the
on-device model. The fetched bytes are processed in memory and are **not**
forwarded anywhere else.

---

## 3. No Telemetry, Analytics, Cloud, or Remote Code

The extension contains **none** of the following:

- ❌ Telemetry or usage tracking
- ❌ Analytics SDKs
- ❌ Cloud inference endpoints
- ❌ Remote code loading
- ❌ Phone-home / beacon calls

All JavaScript and WASM are **bundled** in the extension package. No code
is fetched or evaluated from a remote source at any point.

---

## 4. Settings Storage

User settings (e.g., threshold, calibration parameters, display
preferences) are stored in `chrome.storage.local`. This storage:

- Is **local** to the browser profile.
- Is **not synced** to any cloud account (we do not use
  `chrome.storage.sync`).
- Persists across browser restarts until the user clears it.

No settings data leaves the device.

---

## 5. Detection Results Cache

Detection results are kept in a **bounded in-memory cache** to avoid
re-processing the same image. This cache:

- Lives only in the extension's service-worker / page memory.
- Has a fixed maximum size (entries are evicted on a LRU basis).
- Is **cleared when the browser exits** or the service worker is
  terminated.

Results are **not** written to disk and **not** transmitted anywhere.

---

## 6. Network Can Be Disabled After Model Download

The model file is **downloaded once** on first use and cached in the
browser's Cache Storage. After the model is cached:

- The extension needs **no network access** to perform detection.
- You can place the extension behind a firewall, disable its network
  permissions, or run fully offline — detection continues to work.

The only network activity after initial setup is the extension fetching
image bytes from the page's own origin (see §2), which is required to read
pixels.

---

## 7. Why Broad Host Permissions?

The extension requests broad host permissions (`<all_urls>` or equivalent)
so it can:

- **Read cross-origin `<img>` elements.** Images on the web are served
  from CDNs, third-party domains, and data URIs that differ from the page
  origin. Without broad host permissions, the extension cannot access the
  pixel data of these cross-origin images.
- **Fetch image bytes for local inference.** As described in §2, the
  extension must fetch the image from its hosting domain to read pixels.

These permissions are used **solely** to read image pixels for local,
on-device detection. They are **not** used to read page text, cookies,
form data, or any non-image content. They are **not** used to send any
data anywhere.

---

## Summary

| Concern | Status |
|---|---|
| Where detection happens | On-device (browser WASM/WebGPU) |
| Images uploaded to server | No |
| Telemetry / analytics | None |
| Remote code execution | None |
| Cloud endpoints | None |
| Settings location | `chrome.storage.local` (device only) |
| Results cache | Bounded, in-memory, cleared on exit |
| Network after model download | Optional (can be disabled) |
