/**
 * meta.js — Metadata forensic signal extraction.
 *
 * Scans raw image bytes for EXIF, XMP, PNG tEXt, and other embedded metadata
 * that indicates AI generation (or conversely, authentic camera provenance).
 * `fuseSignals()` blends these hard-evidence signals with the model's
 * probability output.
 */

// ────────────────────────────────────────────────────────────────────────────
// AI software / tool fingerprints (case-insensitive substring match)
// ────────────────────────────────────────────────────────────────────────────

const AI_SOFTWARE = [
  'midjourney', 'stable diffusion', 'automatic1111', 'comfyui',
  'dall-e', 'dalle', 'gpt-4', 'firefly', 'leonardo', 'novelai',
  'niji', 'stability ai', 'runway', 'pika', 'ideogram',
];

const AI_KEYWORDS = [
  'trainedAlgorithmicMedia', 'compositeWithTrainedAlgorithmicMedia',
  'algorithmically-generated', 'ai-generated', 'generative ai',
];

// ────────────────────────────────────────────────────────────────────────────
// Camera software fingerprints (authentic provenance)
// ────────────────────────────────────────────────────────────────────────────

const CAMERA_SOFTWARE = [
  'canon', 'nikon', 'sony', 'fujifilm', 'olympus', 'panasonic',
  'pentax', 'leica', 'apple', 'samsung', 'huawei', 'google',
  'lightroom', 'photoshop', 'capture one', 'dxo',
];

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

const enc = (s) => [...s].map((c) => c.charCodeAt(0));

/**
 * Latin1 byte-decode (sufficient for metadata text we care about).
 * @param {Uint8Array} u8
 * @param {number} start
 * @param {number} end
 * @returns {string}
 */
function bytesToString(u8, start, end) {
  let s = '';
  for (let i = start; i < end; i++) s += String.fromCharCode(u8[i]);
  return s;
}

/**
 * Case-insensitive substring test.
 * @param {string} haystack
 * @param {string} needle
 * @returns {boolean}
 */
function ciIncludes(haystack, needle) {
  return haystack.toLowerCase().includes(needle.toLowerCase());
}

// ────────────────────────────────────────────────────────────────────────────
// PNG parsing
// ────────────────────────────────────────────────────────────────────────────

const PNG_SIG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

function isPng(u8) {
  if (u8.length < 8) return false;
  return PNG_SIG.every((b, i) => u8[i] === b);
}

/**
 * Parse PNG chunks and extract tEXt / iTXt / zTXt keyword-value pairs.
 * @param {Uint8Array} u8
 * @returns {{keyword:string, text:string}[]}
 */
function parsePngText(u8) {
  const results = [];
  let off = 8; // skip signature

  while (off + 8 <= u8.length) {
    const len = (u8[off] << 24) | (u8[off + 1] << 16) | (u8[off + 2] << 8) | u8[off + 3];
    const type = bytesToString(u8, off + 4, off + 8);

    if (type === 'IEND') break;

    const dataStart = off + 8;
    const dataEnd = Math.min(dataStart + len, u8.length);

    if (type === 'tEXt') {
      // Null-separated keyword + text
      const raw = bytesToString(u8, dataStart, dataEnd);
      const nul = raw.indexOf('\0');
      if (nul >= 0) {
        results.push({ keyword: raw.slice(0, nul), text: raw.slice(nul + 1) });
      }
    } else if (type === 'iTXt') {
      // iTXt: keyword\0 compressionFlag(1) compressionMethod(1) langTag\0 translatedKeyword\0 text
      const raw = u8.slice(dataStart, dataEnd);
      const nul1 = raw.indexOf(0);
      if (nul1 >= 0) {
        const keyword = bytesToString(raw, 0, nul1);
        // Skip compression flag(1) + method(1) + langTag\0 + translatedKeyword\0
        let p = nul1 + 3;
        const nul2 = raw.indexOf(0, p);
        if (nul2 >= 0) {
          p = nul2 + 1;
          const nul3 = raw.indexOf(0, p);
          if (nul3 >= 0) {
            const text = bytesToString(raw, nul3 + 1, raw.length);
            results.push({ keyword, text });
          }
        }
      }
    }

    off = dataEnd + 4; // skip CRC
  }

  return results;
}

// ────────────────────────────────────────────────────────────────────────────
// JPEG / EXIF parsing
// ────────────────────────────────────────────────────────────────────────────

function isJpeg(u8) {
  return u8.length >= 2 && u8[0] === 0xff && u8[1] === 0xd8;
}

/**
 * Extract EXIF Software tag (0x0131) and XMP packet from JPEG APP1 segments.
 * @param {Uint8Array} u8
 * @returns {{software:string|null, xmp:string|null}}
 */
function parseJpegMeta(u8) {
  let software = null;
  let xmp = null;

  let off = 2; // skip SOI marker

  while (off + 4 < u8.length) {
    if (u8[off] !== 0xff) break;
    const marker = u8[off + 1];

    // SOS or EOI — stop scanning
    if (marker === 0xda || marker === 0xd9) break;

    // Only process APPn markers (0xe0–0xef)
    if (marker < 0xe0 || marker > 0xef) {
      off += 2;
      continue;
    }

    const segLen = (u8[off + 2] << 8) | u8[off + 3];
    const segStart = off + 4;
    const segEnd = Math.min(off + 2 + segLen, u8.length);

    // Check for EXIF header
    if (segEnd - segStart >= 6 &&
        u8[segStart] === 0x45 && u8[segStart + 1] === 0x78 &&
        u8[segStart + 2] === 0x69 && u8[segStart + 3] === 0x66 &&
        u8[segStart + 4] === 0x00 && u8[segStart + 5] === 0x00) {
      software = parseExifSoftware(u8, segStart + 6, segEnd);
    }

    // Check for XMP header
    const segText = bytesToString(u8, segStart, Math.min(segStart + 35, segEnd));
    if (ciIncludes(segText, 'ns.adobe.com/xap/1.0/')) {
      const nul = u8.indexOf(0, segStart);
      if (nul >= 0 && nul < segEnd) {
        xmp = bytesToString(u8, nul + 1, segEnd);
      }
    }

    off = segEnd;
  }

  return { software, xmp };
}

/**
 * Parse TIFF IFD to find the Software tag (0x0131).
 * @param {Uint8Array} u8
 * @param {number} tiffStart
 * @param {number} tiffEnd
 * @returns {string|null}
 */
function parseExifSoftware(u8, tiffStart, tiffEnd) {
  if (tiffEnd - tiffStart < 8) return null;

  const little = u8[tiffStart] === 0x49; // 'I' = little-endian
  const u16 = (off) => little
    ? u8[off] | (u8[off + 1] << 8)
    : (u8[off] << 8) | u8[off + 1];
  const u32 = (off) => little
    ? u8[off] | (u8[off + 1] << 8) | (u8[off + 2] << 16) | (u8[off + 3] << 24)
    : (u8[off] << 24) | (u8[off + 1] << 16) | (u8[off + 2] << 8) | u8[off + 3];

  const ifdOffset = tiffStart + u32(tiffStart + 4);
  if (ifdOffset + 2 > tiffEnd) return null;

  const entryCount = u16(ifdOffset);
  let p = ifdOffset + 2;

  for (let i = 0; i < entryCount; i++) {
    if (p + 12 > tiffEnd) break;
    const tag = u16(p);
    const type = u16(p + 2);
    const count = u32(p + 4);

    // Tag 0x0131 = Software, type 2 = ASCII
    if (tag === 0x0131 && type === 2 && count > 0) {
      const valOffset = u32(p + 8);
      // If count ≤ 4, value is inline in the offset field
      const dataStart = count <= 4 ? p + 8 : tiffStart + valOffset;
      const dataEnd = Math.min(dataStart + count, tiffEnd);
      let s = bytesToString(u8, dataStart, dataEnd);
      // Strip trailing null
      const nul = s.indexOf('\0');
      if (nul >= 0) s = s.slice(0, nul);
      return s;
    }

    p += 12;
  }

  return null;
}

// ────────────────────────────────────────────────────────────────────────────
// Signal classification
// ────────────────────────────────────────────────────────────────────────────

/**
 * Classify metadata evidence into discrete signals.
 *
 * @param {Uint8Array} u8 - Raw image bytes.
 * @returns {{signals:{id:string, kind:'ai'|'real', label:string, detail:string}[]}}
 */
export function scanMetadata(u8) {
  const signals = [];

  // ── PNG metadata ─────────────────────────────────────────────────────
  if (isPng(u8)) {
    const chunks = parsePngText(u8);

    for (const { keyword, text } of chunks) {
      const kw = keyword.toLowerCase();

      // Stable Diffusion / A1111 parameters
      if (kw === 'parameters') {
        signals.push({
          id: 'sd-parameters',
          kind: 'ai',
          label: 'Stable Diffusion parameters',
          detail: text.slice(0, 200),
        });
      }

      // ComfyUI workflow
      if (kw === 'workflow' || kw === 'prompt') {
        const isJson = text.trimStart().startsWith('{');
        if (isJson || ciIncludes(text, 'class_type') || ciIncludes(text, 'KSampler')) {
          signals.push({
            id: 'comfyui-workflow',
            kind: 'ai',
            label: 'ComfyUI workflow',
            detail: text.slice(0, 200),
          });
        }
      }

      // Generic AI keyword in any PNG text
      for (const ak of AI_KEYWORDS) {
        if (ciIncludes(text, ak)) {
          signals.push({
            id: 'png-ai-keyword',
            kind: 'ai',
            label: `AI keyword in PNG: ${keyword}`,
            detail: ak,
          });
          break;
        }
      }
    }
  }

  // ── JPEG EXIF + XMP ──────────────────────────────────────────────────
  if (isJpeg(u8)) {
    const { software, xmp } = parseJpegMeta(u8);

    if (software) {
      const swLower = software.toLowerCase();

      // AI software
      for (const ai of AI_SOFTWARE) {
        if (swLower.includes(ai)) {
          signals.push({
            id: 'exif-ai-software',
            kind: 'ai',
            label: `EXIF Software: ${software}`,
            detail: ai,
          });
          break;
        }
      }

      // Camera software (authentic provenance)
      if (!signals.some((s) => s.id === 'exif-ai-software')) {
        for (const cam of CAMERA_SOFTWARE) {
          if (swLower.includes(cam)) {
            signals.push({
              id: 'camera-exif',
              kind: 'real',
              label: `Camera software: ${software}`,
              detail: cam,
            });
            break;
          }
        }
      }
    }

    if (xmp) {
      // IPTC digitalSourceType AI indicators
      if (ciIncludes(xmp, 'trainedAlgorithmicMedia')) {
        signals.push({
          id: 'iptc-ai',
          kind: 'ai',
          label: 'IPTC: trainedAlgorithmicMedia',
          detail: 'http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia',
        });
      } else if (ciIncludes(xmp, 'compositeWithTrainedAlgorithmicMedia')) {
        signals.push({
          id: 'iptc-ai',
          kind: 'ai',
          label: 'IPTC: compositeWithTrainedAlgorithmicMedia',
          detail: 'composite with AI',
        });
      }

      // Generic AI keywords in XMP
      for (const ak of AI_KEYWORDS) {
        if (ciIncludes(xmp, ak) && !signals.some((s) => s.id === 'iptc-ai')) {
          signals.push({
            id: 'xmp-ai-keyword',
            kind: 'ai',
            label: `XMP AI keyword`,
            detail: ak,
          });
          break;
        }
      }
    }
  }

  return { signals };
}

// ────────────────────────────────────────────────────────────────────────────
// Signal fusion
// ────────────────────────────────────────────────────────────────────────────

/**
 * Fuse metadata signals with the model probability.
 *
 * Rules:
 *   - Hard AI evidence (sd-parameters, comfyui-workflow, exif-ai-software,
 *     iptc-ai) → raise to at least 0.97, but never lower an already-higher
 *     probability.
 *   - Camera EXIF (real provenance) → no change (model probability stands;
 *     the frequency and model signals are more decisive).
 *   - No signals → return p unchanged.
 *
 * @param {number} p - Current fused probability (from model + frequency).
 * @param {{id:string, kind:string, label:string, detail:string}[]} signals
 * @returns {number} Adjusted probability.
 */
export function fuseSignals(p, signals) {
  const hasHardAi = signals.some(
    (s) =>
      s.kind === 'ai' &&
      ['sd-parameters', 'comfyui-workflow', 'exif-ai-software', 'iptc-ai'].includes(s.id),
  );

  if (hasHardAi) {
    return Math.max(p, 0.97);
  }

  // No hard evidence — model + frequency probability stands.
  return p;
}
