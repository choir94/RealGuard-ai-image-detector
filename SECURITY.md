# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.x     | ✅ Security updates |
| < 2.0   | ❌ Not supported    |

## Model Integrity — SHA-256 Verification

The ONNX model file is verified at runtime against a pinned SHA-256 hash to
detect tampering or corruption:

```js
// Computed once, stored in source — do not generate at runtime.
const EXPECTED_SHA256 = "a42c7d740fbb345ba9a26d469b22f301d73089ce3c6da993877ed2b6965a8ba1";

const buf = await file.arrayBuffer();
const digest = await crypto.subtle.digest("SHA-256", buf);
const hash = [...new Uint8Array(digest)]
  .map(b => b.toString(16).padStart(2, "0"))
  .join("");

if (hash !== EXPECTED_SHA256) {
  throw new Error("Model integrity check failed — refusing to load.");
}
```

If the hash does not match, the model is **not** loaded. Treat any mismatch as
a potential supply-chain compromise.

## No Remote Code Execution

This project adheres to the following constraints:

- **No `eval()` or `Function()` constructor** — no dynamic code execution.
- **No remote script loading** — no `<script src>` to external origins, no
  dynamic import of remote modules.
- **Model runs locally only** — inference happens entirely in the browser via
  ONNX Runtime Web (WASM). No image data is uploaded to any server.
- **No automatic model updates** — the model is bundled at build time. A hash
  change requires a new release.

## Content Security Policy (CSP)

The extension manifest declares the following CSP for extension pages:

```
script-src 'self' 'wasm-unsafe-eval'; object-src 'self'
```

- `'wasm-unsafe-eval'` is required **only** for ONNX Runtime WASM execution;
  it does not enable JavaScript `eval()`.
- `'self'` restricts scripts and objects to the extension's own packaged files.

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

1. Open a **private security advisory** on GitHub:
   `https://github.com/choir94/RealGuard-ai-image-detector/security/advisories/new`
2. Include a clear description, reproduction steps, and impact assessment.
3. You will receive an acknowledgment within **48 hours**.
4. A fix or mitigation timeline will be provided within **7 days**.

Please allow reasonable time for a fix before public disclosure. We credit
responsible reporters in release notes unless they prefer to remain anonymous.
