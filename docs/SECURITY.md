# Security Policy

This document describes the security measures built into the AI Image
Detector extension and how to report vulnerabilities.

---

## 1. Model Integrity Verification (SHA-256)

Every model download is verified by **SHA-256 hash** before it is accepted
into cache.

- **Pinned hash:**
  `a42c7d740fbb345ba9a26d469b22f301d73089ce3c6da993877ed2b6965a8ba1`
- The extension computes the SHA-256 of the downloaded bytes and compares
  it to the pinned hash.
- If the hash **does not match**, the download is **rejected** and the
  model is not loaded. The extension surfaces an error and refuses to
  run inference rather than executing an untrusted model.
- This protects against man-in-the-middle attacks, CDN compromise, and
  accidental corruption.

---

## 2. No Remote Code Execution

**All JavaScript and WASM are bundled** in the extension package.

- ❌ No `eval()`
- ❌ No `Function()` constructor
- ❌ No dynamic `import()` of remote URLs
- ❌ No `setTimeout`/`setInterval` with string arguments
- ❌ No fetching of scripts from remote origins
- ❌ No `document.write()` of remote content

The only remote artifact fetched at runtime is the **model weights file**
(see §4), which is a binary data blob — never executed as code.

---

## 3. Content Security Policy (CSP)

The extension enforces a strict Content Security Policy:

```
script-src 'self' 'wasm-unsafe-eval';
```

| Directive | Meaning |
|---|---|
| `'self'` | Scripts may only load from the extension's own bundled files |
| `'wasm-unsafe-eval'` | Allows WebAssembly compilation (required by ONNX Runtime Web) — does **not** allow JS `eval()` |

Notably **absent** from the CSP:

- ❌ `'unsafe-eval'` — JavaScript `eval()` is blocked
- ❌ `'unsafe-inline'` — inline scripts are blocked
- ❌ Any `https://` source — no remote script origins are allowed

This means even if an attacker could inject HTML into an extension page,
they could not execute arbitrary JavaScript.

---

## 4. Model Download & Cache Verification

The model lifecycle is designed for **download-once, verify-always**:

1. **First run.** The model is downloaded from the pinned Hugging Face URL.
2. **Hash check on download.** SHA-256 is computed and compared to the
   pinned value. Mismatch → reject.
3. **Cached in Cache Storage.** The verified model is stored in the
   browser's Cache Storage API (persistent, origin-scoped).
4. **Hash check on every load.** Each time the model is loaded from cache,
   the SHA-256 is re-verified. This detects cache tampering or corruption
   that may have occurred on disk.
5. **No re-download needed.** After the initial download, the model is
   served from cache. Network access is not required for subsequent loads.

```
Download → SHA-256 verify → Cache Storage
                                    ↓
Load from cache → SHA-256 re-verify → ONNX Runtime → Inference
```

---

## 5. Service Worker Constraints

The extension's service worker (background script) adheres to:

- **No `eval()`** — blocked by CSP and absent from code.
- **No dynamic `import()`** — all modules are statically imported and
  bundled at build time.
- **No remote `fetch()` of executable code** — the service worker only
  fetches the model weights blob (binary data, not code).
- **No persistent background connections** to external servers.

---

## 6. No Remote Code Updates

The extension does **not** support hot-loading or remote updating of its
logic. All updates ship through the standard browser extension update
channel (Chrome Web Store / signed package). This prevents an attacker
from injecting new logic at runtime even if they compromise a CDN.

---

## 7. Reporting Vulnerabilities

We take security reports seriously. If you believe you have found a
security vulnerability in the AI Image Detector extension, please report
it responsibly.

### How to Report

1. **Do not** open a public issue for security vulnerabilities.
2. Send a description of the vulnerability to the project maintainer
   through a private channel.
3. Include:
   - A description of the issue and its potential impact.
   - Steps to reproduce (proof of concept if possible).
   - The extension version and browser/OS you tested on.
4. You will receive an acknowledgment within **72 hours**.
5. We will investigate and, if confirmed, work on a fix and coordinate
   disclosure.

### Scope

In scope:

- Bypass of the SHA-256 model integrity check.
- Remote code execution within the extension context.
- CSP bypass leading to arbitrary script execution.
- Exfiltration of image data to a remote endpoint.
- Tampering with the Cache Storage model blob without detection.

Out of scope:

- The model's classification accuracy (false positives / negatives) —
  this is a functionality concern, not a security vulnerability.
- Issues in third-party dependencies that are already publicly disclosed
  and patched in the latest version.

---

## Summary

| Control | Status |
|---|---|
| SHA-256 verification on download | ✅ Enforced |
| SHA-256 re-verification on cache load | ✅ Enforced |
| No remote code execution | ✅ All code bundled |
| CSP `script-src 'self' 'wasm-unsafe-eval'` | ✅ Enforced |
| No `eval()` in service worker | ✅ Blocked by CSP + code |
| No dynamic `import()` in service worker | ✅ Not used |
| Model cached after download | ✅ Cache Storage |
| Network optional after download | ✅ Can be disabled |
