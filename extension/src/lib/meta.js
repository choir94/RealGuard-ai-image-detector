/**
 * meta.js — Metadata forensics module for AI image detection.
 *
 * DESIGN PHILOSOPHY
 * -----------------
 * Metadata can PROVE an image is AI-generated (a signed C2PA manifest that
 * asserts algorithmic media, an embedded Stable Diffusion "parameters" chunk,
 * an IPTC digitalSourceType of trainedAlgorithmicMedia), but the ABSENCE of
 * such metadata proves nothing. Camera EXIF is trivially spoofable. Therefore:
 *
 *   - AI evidence RAISES the final detector score. Hard proof (embedded
 *     generation prompts/workflows, C2PA AI assertions, IPTC/XMP
 *     trainedAlgorithmicMedia) floors the fused score at 0.97.
 *   - Strong-but-not-proof AI indicators (a Software/CreatorTool string that
 *     names a known generator) raise the score to a high-but-not-maximal level.
 *   - "Real" evidence (genuine camera EXIF carrying both a MakerNote and a
 *     DateTimeOriginal) is surfaced as CONTEXT ONLY and never changes the
 *     score — it is informative, not exculpatory, because it can be faked.
 *
 * This module performs no network access and touches no DOM. It operates purely
 * on a Uint8Array of raw image bytes and supports PNG, JPEG, WebP, and TIFF.
 *
 * EXPORTS
 *   scanMetadata(u8: Uint8Array) -> { signals: Signal[] }
 *   fuseSignals(pCal: number, signals: Signal[]) -> number
 */

/* ------------------------------------------------------------------ *
 * Constants
 * ------------------------------------------------------------------ */

/**
 * Lowercase substrings that, when found inside a Software / CreatorTool /
 * Originating Program / C2PA claim-generator field, indicate an AI image
 * generator. Matched case-insensitively as substrings.
 */
const AI_GENERATOR_MARKERS = [
  'midjourney', 'dall-e', 'dall·e', 'dalle', 'openai', 'gpt-4o', 'chatgpt',
  'adobe firefly', 'firefly', 'stable diffusion', 'stablediffusion', 'sdxl',
  'stability ai', 'stability.ai', 'flux', 'black forest labs', 'ideogram',
  'recraft', 'imagen', 'gemini', 'grok', 'aurora', 'novelai', 'comfyui',
  'invokeai', 'automatic1111', 'sd-webui', 'fooocus', 'draw things',
  'leonardo.ai', 'leonardo ai', 'krea', 'luma', 'reve', 'seedream', 'kling',
  'wan2', 'hidream', 'playground ai', 'bing image creator', 'designer.microsoft'
];

/**
 * IPTC digitalSourceType values (lowercased, punctuation-stripped) that denote
 * algorithmically generated / trained-media content.
 */
const IPTC_AI_SOURCETYPES = [
  'trainedalgorithmicmedia',
  'compositewithtrainedalgorithmicmedia',
  'algorithmicmedia'
];

/**
 * Signal IDs that constitute HARD PROOF of AI generation. Any one of these
 * floors the fused probability at 0.97.
 */
const HARD_AI_IDS = new Set([
  'sd-parameters',          // embedded Stable Diffusion / A1111 parameters chunk
  'comfyui-workflow',       // embedded ComfyUI workflow JSON
  'comfyui-prompt',         // embedded ComfyUI prompt (API) JSON
  'novelai',                // NovelAI software tag or Comment JSON
  'c2pa-ai-assertion',      // C2PA manifest asserting trainedAlgorithmicMedia
  'c2pa-generator-ai',      // C2PA claim_generator naming an AI tool
  'iptc-ai'                 // XMP IPTC digitalSourceType = AI media
]);

/**
 * Signal IDs that are STRONG INDICATORS (but not cryptographic proof) of AI
 * generation — typically a spoofable text field that names a generator. These
 * raise the fused score to a high but non-maximal level.
 */
const STRONG_AI_IDS = new Set([
  'png-software-ai', 'png-text-ai',
  'exif-ai-software',
  'xmp-creatortool-ai', 'xmp-tool-ai',
  'iptc-iim-ai', 'iptc-iim-source-ai', 'iptc-iim-credit-ai', 'iptc-iim-byline-ai',
  'c2pa-manifest'
]);

/* ------------------------------------------------------------------ *
 * Small helpers
 * ------------------------------------------------------------------ */

/** Build a signal object. kind is 'ai' | 'real' | 'info'. */
function sig(id, kind, label, detail) {
  return { id, kind, label, detail: detail == null ? '' : String(detail) };
}

/** Truncate a string to n chars, appending an ellipsis if truncated. */
function trim(s, n) {
  if (s == null) return '';
  s = String(s);
  if (s.length <= n) return s;
  return s.slice(0, n) + '…';
}

function lower(s) {
  return s == null ? '' : String(s).toLowerCase();
}

/** Decode a byte range as Latin-1 (1 byte = 1 char). Never throws. */
function decodeLatin1(bytes, start, end) {
  let s = '';
  for (let i = start; i < end; i++) s += String.fromCharCode(bytes[i]);
  return s;
}

/** Decode a byte range as UTF-8, falling back to Latin-1. */
function decodeUtf8(bytes, start, end) {
  try {
    return new TextDecoder('utf-8', { fatal: false }).decode(bytes.subarray(start, end));
  } catch (e) {
    return decodeLatin1(bytes, start, end);
  }
}

/**
 * Return the first AI_GENERATOR_MARKERS substring found in `str`
 * (case-insensitive), or null if none match.
 */
function matchGenerator(str) {
  if (!str) return null;
  const l = lower(str);
  for (let i = 0; i < AI_GENERATOR_MARKERS.length; i++) {
    if (l.indexOf(AI_GENERATOR_MARKERS[i]) !== -1) return AI_GENERATOR_MARKERS[i];
  }
  return null;
}

/* Big-endian readers */
function u16be(u8, o) { return (u8[o] << 8) | u8[o + 1]; }
function u32be(u8, o) {
  return ((u8[o] << 24) | (u8[o + 1] << 16) | (u8[o + 2] << 8) | u8[o + 3]) >>> 0;
}
/* Little-endian readers */
function u16le(u8, o) { return u8[o] | (u8[o + 1] << 8); }
function u32le(u8, o) {
  return (u8[o] | (u8[o + 1] << 8) | (u8[o + 2] << 16) | (u8[o + 3] << 24)) >>> 0;
}
/* Endian-parameterised readers (TIFF) */
function u16(u8, o, le) { return le ? u16le(u8, o) : u16be(u8, o); }
function u32(u8, o, le) { return le ? u32le(u8, o) : u32be(u8, o); }

/** Read an EXIF ASCII field: stops at the first NUL, replaces unprintables. */
function readAscii(u8, start, end) {
  let s = '';
  for (let i = start; i < end; i++) {
    const c = u8[i];
    if (c === 0) break;
    s += (c < 32 || c > 126) ? ' ' : String.fromCharCode(c);
  }
  return s.trim();
}

/** Index of the first NUL byte in [start, end), or -1. */
function findNull(u8, start, end) {
  for (let i = start; i < end; i++) if (u8[i] === 0) return i;
  return -1;
}

/**
 * Extract printable ASCII runs of length >= minLen from a byte range.
 * Used for scanning opaque binary blobs (C2PA/JUMBF) for human-readable strings.
 */
function extractStrings(u8, start, end, minLen) {
  const out = [];
  let cur = '';
  for (let i = start; i < end; i++) {
    const c = u8[i];
    if (c >= 32 && c < 127) cur += String.fromCharCode(c);
    else {
      if (cur.length >= minLen) out.push(cur);
      cur = '';
    }
  }
  if (cur.length >= minLen) out.push(cur);
  return out;
}

/**
 * Inflate a zlib stream (zTXt / compressed iTXt). Prefers pako when present.
 * NOTE: the dominant AI generators (A1111, ComfyUI, NovelAI) emit UNCOMPRESSED
 * tEXt chunks, so the inability to inflate only affects the rarer zTXt /
 * compressed-iTXt case — those chunks are then simply skipped.
 */
function inflateZlib(data) {
  if (typeof pako !== 'undefined' && pako && typeof pako.inflate === 'function') {
    try { return pako.inflate(data); } catch (e) { return null; }
  }
  return null;
}

/* ------------------------------------------------------------------ *
 * Format detection
 * ------------------------------------------------------------------ */

const PNG_SIG = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

function isJpeg(u8) { return u8.length >= 2 && u8[0] === 0xff && u8[1] === 0xd8; }

function isPng(u8) {
  if (u8.length < 8) return false;
  for (let i = 0; i < 8; i++) if (u8[i] !== PNG_SIG[i]) return false;
  return true;
}

function isWebp(u8) {
  return u8.length >= 12 &&
    u8[0] === 0x52 && u8[1] === 0x49 && u8[2] === 0x46 && u8[3] === 0x46 && // RIFF
    u8[8] === 0x57 && u8[9] === 0x45 && u8[10] === 0x42 && u8[11] === 0x50;   // WEBP
}

function isTiff(u8) {
  if (u8.length < 4) return false;
  return (u8[0] === 0x49 && u8[1] === 0x49 && u8[2] === 0x2a && u8[3] === 0x00) || // II* (LE)
         (u8[0] === 0x4d && u8[1] === 0x4d && u8[2] === 0x00 && u8[3] === 0x2a);    // MM* (BE)
}

/* ------------------------------------------------------------------ *
 * PNG — tEXt / iTXt / zTXt chunks
 * ------------------------------------------------------------------ */

function scanPng(u8, signals) {
  let p = 8; // skip signature
  while (p + 12 <= u8.length) {
    const len = u32be(u8, p);
    const type = decodeLatin1(u8, p + 4, p + 8);
    const dataStart = p + 8;
    const dataEnd = dataStart + len;
    if (dataEnd + 4 > u8.length) break; // truncated

    if (type === 'tEXt' || type === 'zTXt' || type === 'iTXt') {
      parsePngText(u8, dataStart, len, type, signals);
    }
    if (type === 'IEND') break;
    p = dataEnd + 4; // skip data + CRC
  }
}

function parsePngText(u8, start, len, type, signals) {
  const end = start + len;

  // keyword is null-terminated Latin-1
  let kEnd = start;
  while (kEnd < end && u8[kEnd] !== 0) kEnd++;
  if (kEnd >= end) return;
  const keyword = decodeLatin1(u8, start, kEnd);

  let text = '';
  if (type === 'tEXt') {
    text = decodeUtf8(u8, kEnd + 1, end);
  } else if (type === 'zTXt') {
    // layout: keyword \0 compression_method(1) compressed_text
    if (kEnd + 2 < end && u8[kEnd + 1] === 0) { // method 0 = zlib/deflate
      const inflated = inflateZlib(u8.subarray(kEnd + 2, end));
      if (inflated) text = decodeUtf8(inflated, 0, inflated.length);
    }
  } else if (type === 'iTXt') {
    // layout: keyword \0 comp_flag(1) comp_method(1) lang \0 trans_kw \0 text
    if (kEnd + 3 > end) return;
    const compFlag = u8[kEnd + 1];
    const compMethod = u8[kEnd + 2];
    let i = kEnd + 3;
    // skip language tag
    while (i < end && u8[i] !== 0) i++;
    i++; // skip NUL
    // skip translated keyword
    while (i < end && u8[i] !== 0) i++;
    i++; // skip NUL
    if (i <= end) {
      if (compFlag === 1 && compMethod === 0) {
        const inflated = inflateZlib(u8.subarray(i, end));
        if (inflated) text = decodeUtf8(inflated, 0, inflated.length);
      } else {
        text = decodeUtf8(u8, i, end);
      }
    }
  }

  const kw = lower(keyword);

  // --- Stable Diffusion / A1111 parameters ---
  // Canonical A1111 format: prompt\nNegative prompt: ...\nSteps: N, Sampler: ..., CFG scale: ...
  if (kw === 'parameters') {
    const isSd = /steps:\s*\d+/i.test(text) &&
                 (/sampler:/i.test(text) || /cfg scale:/i.test(text));
    if (isSd) {
      signals.push(sig('sd-parameters', 'ai',
        'Embedded Stable Diffusion parameters',
        trim(text.replace(/\s+/g, ' '), 220)));
    }
    return;
  }

  // --- ComfyUI workflow / prompt (JSON) ---
  if (kw === 'workflow') {
    // Editor-format workflow: contains "nodes" and "links" arrays.
    // API-format prompt: contains "class_type" + "inputs" per node.
    if (/"nodes"\s*:/.test(text) && /"links"\s*:/.test(text)) {
      signals.push(sig('comfyui-workflow', 'ai',
        'Embedded ComfyUI workflow',
        trim(text.replace(/\s+/g, ' '), 220)));
    } else if (/class_type/i.test(text) && /"inputs"/i.test(text)) {
      signals.push(sig('comfyui-workflow', 'ai',
        'Embedded ComfyUI workflow',
        trim(text.replace(/\s+/g, ' '), 220)));
    }
    return;
  }
  if (kw === 'prompt') {
    if (/class_type/i.test(text) && /"inputs"/i.test(text)) {
      signals.push(sig('comfyui-prompt', 'ai',
        'Embedded ComfyUI prompt graph',
        trim(text.replace(/\s+/g, ' '), 220)));
    }
    return;
  }

  // --- NovelAI: software tag + Comment JSON ---
  if (kw === 'software') {
    const m = matchGenerator(text);
    if (lower(text).indexOf('novelai') !== -1) {
      signals.push(sig('novelai', 'ai',
        'NovelAI software tag', trim(text, 80)));
    } else if (m) {
      signals.push(sig('png-software-ai', 'ai',
        'PNG software: AI generator (' + m + ')', trim(text, 80)));
    } else if (text) {
      signals.push(sig('png-software', 'info', 'PNG software tag', trim(text, 80)));
    }
    return;
  }
  if (kw === 'comment') {
    // NovelAI stores a JSON object with sampler/steps/seed/scale in 'Comment'.
    if (/^\s*\{/.test(text) && /"sampler"\s*:/.test(text) &&
        /"steps"\s*:/.test(text) && (/"seed"\s*:/.test(text) || /"scale"\s*:/.test(text))) {
      signals.push(sig('novelai', 'ai',
        'NovelAI generation parameters (Comment)',
        trim(text.replace(/\s+/g, ' '), 220)));
    }
    return;
  }

  // --- Adobe XMP packet carried in a PNG tEXt chunk ---
  if (kw === 'xml:com.adobe.xmp') {
    parseXmp(text, signals, 'png');
    return;
  }

  // --- Generic fallback: a description/comment chunk naming a generator ---
  if (text && (kw.indexOf('description') !== -1 || kw.indexOf('comment') !== -1)) {
    const m = matchGenerator(text);
    if (m) {
      signals.push(sig('png-text-ai', 'ai',
        'PNG ' + keyword + ': AI marker (' + m + ')', trim(text, 100)));
    }
  }
}

/* ------------------------------------------------------------------ *
 * JPEG — APP1 (EXIF / XMP), APP11 (JUMBF/C2PA), APP13 (Photoshop IRB)
 * ------------------------------------------------------------------ */

function scanJpeg(u8, signals) {
  let p = 2; // after SOI (FF D8)
  while (p + 4 <= u8.length) {
    if (u8[p] !== 0xff) break;
    const marker = u8[p + 1];

    // Standalone markers carry no length field.
    if (marker === 0xd8 || marker === 0xd9 ||        // SOI / EOI
        (marker >= 0xd0 && marker <= 0xd7) ||         // RSTn
        marker === 0x01) {                             // TEM
      p += 2;
      continue;
    }
    if (p + 4 > u8.length) break;
    const segLen = u16be(u8, p + 2);
    if (segLen < 2) break;
    const segStart = p + 4;
    const segEnd = p + 2 + segLen;
    if (segEnd > u8.length) break;

    if (marker === 0xe1) {            // APP1
      handleApp1(u8, segStart, segEnd, signals);
    } else if (marker === 0xeb) {     // APP11 — JUMBF / C2PA
      scanJumbf(u8, segStart, segEnd, signals);
    } else if (marker === 0xed) {     // APP13 — Photoshop IRB / IPTC-IIM
      parsePhotoshopIrb(u8, segStart, segEnd, signals);
    } else if (marker === 0xda) {     // SOS — entropy-coded image data follows
      break;
    }

    p = segEnd;
  }
}

function handleApp1(u8, segStart, segEnd, signals) {
  // EXIF:  "Exif\0\0" + TIFF header
  if (segEnd - segStart >= 6 &&
      u8[segStart] === 0x45 && u8[segStart + 1] === 0x78 &&
      u8[segStart + 2] === 0x69 && u8[segStart + 3] === 0x66 &&
      u8[segStart + 4] === 0x00 && u8[segStart + 5] === 0x00) {
    parseExifTiff(u8, segStart + 6, segEnd, signals, 'jpeg');
    return;
  }
  // XMP: "http://ns.adobe.com/xap/1.0/\0" + XMP packet
  const head = decodeLatin1(u8, segStart, Math.min(segStart + 29, segEnd));
  if (head.indexOf('http://ns.adobe.com/xap/1.0/') === 0) {
    const nul = findNull(u8, segStart, segEnd);
    const xmpStart = nul >= 0 ? nul + 1 : segStart + 29;
    parseXmp(decodeUtf8(u8, xmpStart, segEnd), signals, 'jpeg');
    return;
  }
  // Extended XMP ("http://ns.adobe.com/xmp/extension/\0") is intentionally
  // skipped — reassembling multi-segment extended packets is out of scope.
}

/* ---- APP13 Photoshop IRB + IPTC-IIM ---- */

function parsePhotoshopIrb(u8, start, end, signals) {
  const head = decodeLatin1(u8, start, Math.min(start + 14, end));
  if (head.indexOf('Photoshop 3.0') !== 0) return;

  let p = start + 14;
  while (p + 12 <= end) {
    const tag = decodeLatin1(u8, p, p + 4);
    if (tag !== '8BIM') { p++; continue; }

    const resId = u16be(u8, p + 4);
    const nameLen = u8[p + 6];
    let nameField = 1 + nameLen;
    if (nameField % 2 !== 0) nameField++;       // pascal name padded to even
    const sizeOff = p + 6 + nameField;
    if (sizeOff + 4 > end) break;
    const dataSize = u32be(u8, sizeOff);
    const dataStart = sizeOff + 4;
    if (dataStart + dataSize > end) break;

    if (resId === 0x0404) {                      // IPTC-IIM resource
      parseIptcIim(u8, dataStart, dataStart + dataSize, signals);
    }

    let next = dataStart + dataSize;
    if (dataSize % 2 !== 0) next++;              // data padded to even
    if (next <= p) break;                         // guard against stall
    p = next;
  }
}

function parseIptcIim(u8, start, end, signals) {
  // IIM records: 0x1C record dataset size(2) data
  const fields = Object.create(null);
  let p = start;
  while (p + 5 <= end) {
    if (u8[p] !== 0x1c) { p++; continue; }
    const rec = u8[p + 1];
    const ds = u8[p + 2];
    const dlen = u16be(u8, p + 3);
    const dstart = p + 5;
    const dend = Math.min(dstart + dlen, end);
    if (rec === 2) {
      const txt = decodeLatin1(u8, dstart, dend).trim();
      if (txt) {
        const key = ds; // dataset number within record 2
        (fields[key] || (fields[key] = [])).push(txt);
      }
    }
    p = dlen > 0 ? dend : p + 5;
  }

  // 2#65 Originating Program, 2#80 By-line, 2#110 Credit, 2#115 Source
  const checks = [
    ['65', 'Originating Program', 'iptc-iim-ai'],
    ['80', 'By-line', 'iptc-iim-byline-ai'],
    ['110', 'Credit', 'iptc-iim-credit-ai'],
    ['115', 'Source', 'iptc-iim-source-ai']
  ];
  for (const [ds, label, id] of checks) {
    const val = fields[ds] ? fields[ds].join('; ') : null;
    if (!val) continue;
    const m = matchGenerator(val);
    if (m) {
      signals.push(sig(id, 'ai', 'IPTC ' + label + ': AI generator (' + m + ')',
        trim(val, 80)));
    }
  }
}

/* ------------------------------------------------------------------ *
 * TIFF / EXIF IFD parsing (used by JPEG APP1, WebP EXIF, and raw TIFF)
 * ------------------------------------------------------------------ */

/**
 * Parse a TIFF structure beginning at `tiffStart`. Collects tags across IFD0,
 * ExifIFD, and the next-IFD chain into a context object, then interprets it.
 */
function parseExifTiff(u8, tiffStart, tiffEnd, signals, source) {
  if (tiffEnd - tiffStart < 8) return;
  const bo = u8[tiffStart];
  let le;
  if (bo === 0x49) le = true;        // 'II' little-endian
  else if (bo === 0x4d) le = false;  // 'MM' big-endian
  else return;

  const magic = u16(u8, tiffStart + 2, le);
  if (magic !== 0x002a) return;       // only classic TIFF (BigTIFF 0x002B unsupported)

  const ctx = {
    software: null, make: null, model: null,
    dateTimeOriginal: null, makerNote: false
  };
  const ifd0Offset = u32(u8, tiffStart + 4, le);
  parseIfdEntries(u8, tiffStart, tiffStart + ifd0Offset, tiffEnd, le, ctx, 0);
  interpretExif(ctx, signals, source);
}

function parseIfdEntries(u8, tiffStart, ifdPos, tiffEnd, le, ctx, depth) {
  if (depth > 4) return;                       // recursion guard
  if (ifdPos < tiffStart || ifdPos + 2 > tiffEnd) return;

  const count = u16(u8, ifdPos, le);
  let p = ifdPos + 2;
  let exifIfdOffset = 0;

  for (let i = 0; i < count; i++) {
    if (p + 12 > tiffEnd) break;
    const tag = u16(u8, p, le);
    const type = u16(u8, p + 2, le);
    const cnt = u32(u8, p + 4, le);
    const valField = p + 8;

    let strVal = null;
    if (type === 2 && cnt > 0) {               // ASCII
      let soff = valField;
      if (cnt > 4) soff = tiffStart + u32(u8, valField, le);
      if (soff >= tiffStart && soff < tiffEnd) {
        strVal = readAscii(u8, soff, Math.min(soff + cnt, tiffEnd));
      }
    }

    switch (tag) {
      case 0x010f: if (!ctx.make) ctx.make = strVal; break;             // Make
      case 0x0110: if (!ctx.model) ctx.model = strVal; break;           // Model
      case 0x0131: if (!ctx.software) ctx.software = strVal; break;     // Software
      case 0x8769: exifIfdOffset = u32(u8, valField, le); break;        // ExifIFD
      case 0x9003: if (!ctx.dateTimeOriginal) ctx.dateTimeOriginal = strVal; break; // DateTimeOriginal
      case 0x927c: ctx.makerNote = true; break;                         // MakerNote (presence)
    }
    p += 12;
  }

  // Next IFD offset (immediately after the entries).
  let nextIfd = 0;
  if (p + 4 <= tiffEnd) nextIfd = u32(u8, p, le);

  if (exifIfdOffset) {
    parseIfdEntries(u8, tiffStart, tiffStart + exifIfdOffset, tiffEnd, le, ctx, depth + 1);
  }
  if (nextIfd) {
    parseIfdEntries(u8, tiffStart, tiffStart + nextIfd, tiffEnd, le, ctx, depth + 1);
  }
}

function interpretExif(ctx, signals, source) {
  // Software — strong AI signal if it names a generator.
  if (ctx.software) {
    const m = matchGenerator(ctx.software);
    if (m) {
      signals.push(sig('exif-ai-software', 'ai',
        'EXIF Software: AI generator',
        trim(ctx.software + ' [' + m + ']', 100)));
    } else {
      signals.push(sig('exif-software', 'info',
        'EXIF Software', trim(ctx.software, 80)));
    }
  }

  // Camera make/model — context only (spoofable).
  if (ctx.make || ctx.model) {
    signals.push(sig('exif-camera', 'info',
      'EXIF camera', trim((ctx.make || '') + ' ' + (ctx.model || ''), 60)));
  }

  if (ctx.dateTimeOriginal) {
    signals.push(sig('exif-datetime', 'info',
      'EXIF DateTimeOriginal', trim(ctx.dateTimeOriginal, 40)));
  }

  // Genuine-looking camera provenance: MakerNote + DateTimeOriginal together.
  // Surfaced as 'real' context — informative, never score-changing.
  if (ctx.makerNote && ctx.dateTimeOriginal) {
    signals.push(sig('camera-exif', 'real',
      'Genuine-looking camera EXIF',
      'MakerNote + DateTimeOriginal present (context only)'));
  }
}

/* ------------------------------------------------------------------ *
 * C2PA / JUMBF manifest scanning (APP11)
 * ------------------------------------------------------------------ */

function scanJumbf(u8, start, end, signals) {
  // A JUMBF box is identified by a 4-byte type code 'jumd' (0x6a756d64),
  // typically found 4 bytes into a box (after the LBox).
  let isJumbf = false;
  for (let i = start; i + 8 <= end; i++) {
    if (u8[i + 4] === 0x6a && u8[i + 5] === 0x75 &&
        u8[i + 6] === 0x6d && u8[i + 7] === 0x64) {
      isJumbf = true;
      break;
    }
  }

  const strings = extractStrings(u8, start, end, 4);
  const joined = strings.join(' ');
  const lowerJoined = lower(joined);

  const hasC2pa = lowerJoined.indexOf('c2pa') !== -1;
  if (!isJumbf && !hasC2pa) return;

  // Claim-generator string (e.g. "com.adobe.c2pa.photo", a tool name, or a URI).
  let claimGen = null;
  for (const s of strings) {
    const ls = lower(s);
    if (ls.indexOf('claim_generator') !== -1 || ls.indexOf('claimgenerator') !== -1) {
      claimGen = s;
      break;
    }
  }

  // AI-generation assertion inside the manifest.
  const aiAssertion =
    lowerJoined.indexOf('trainedalgorithmicmedia') !== -1 ||
    lowerJoined.indexOf('compositewithtrainedalgorithmicmedia') !== -1 ||
    lowerJoined.indexOf('algorithmicmedia') !== -1;

  // The manifest itself exists — record it (info unless it asserts AI).
  let detail = 'C2PA/JUMBF manifest detected';
  if (claimGen) detail += '; claim_generator: ' + trim(claimGen, 80);
  signals.push(sig('c2pa-manifest', hasC2pa || aiAssertion ? 'ai' : 'info',
    'C2PA manifest present', detail));

  if (aiAssertion) {
    signals.push(sig('c2pa-ai-assertion', 'ai',
      'C2PA AI-generation assertion',
      'trainedAlgorithmicMedia / algorithmicMedia asserted in manifest'));
  }

  // Does the claim generator name a known AI tool?
  const m = matchGenerator(joined);
  if (m) {
    signals.push(sig('c2pa-generator-ai', 'ai',
      'C2PA claim generator: AI tool (' + m + ')',
      trim(claimGen || joined, 120)));
  }
}

/* ------------------------------------------------------------------ *
 * XMP parsing
 * ------------------------------------------------------------------ */

function parseXmp(xml, signals, source) {
  if (!xml || xml.length < 4) return;
  const lx = lower(xml);

  // 1. IPTC digitalSourceType = AI media.
  if (lx.indexOf('digitalsourcetype') !== -1) {
    for (const st of IPTC_AI_SOURCETYPES) {
      if (lx.indexOf(st) !== -1) {
        signals.push(sig('iptc-ai', 'ai',
          'XMP: IPTC digitalSourceType = AI media', st));
      }
    }
  }

  // 2. CreatorTool / Software / Source fields naming a generator.
  //    Attribute form:  xmp:CreatorTool="DALL·E 3"
  //    Element form:    <xmp:CreatorTool>DALL·E 3</xmp:CreatorTool>
  const toolValues = [];

  const attrRe = /(?:xmp:CreatorTool|creatorTool|photoshop:Source|tiff:Software|exif:Software|aux:Software)\s*=\s*"([^"]*)"/gi;
  let m;
  while ((m = attrRe.exec(xml)) !== null) toolValues.push(m[1].trim());

  const elemRe = /<(?:xmp|photoshop|tiff|exif|aux):(?:CreatorTool|Software|Source)[^>]*>([^<]*)</gi;
  while ((m = elemRe.exec(xml)) !== null) toolValues.push(m[1].trim());

  for (const v of toolValues) {
    if (!v) continue;
    const gm = matchGenerator(v);
    if (gm) {
      signals.push(sig('xmp-creatortool-ai', 'ai',
        'XMP CreatorTool/Software: AI generator',
        trim(v + ' [' + gm + ']', 100)));
    }
  }
}

/* ------------------------------------------------------------------ *
 * WebP — RIFF chunk parsing (EXIF + XMP chunks)
 * ------------------------------------------------------------------ */

function scanWebp(u8, signals) {
  let p = 12; // skip 'RIFF' + size + 'WEBP'
  while (p + 8 <= u8.length) {
    const fourcc = decodeLatin1(u8, p, p + 4);
    const size = u32le(u8, p + 4);
    const dataStart = p + 8;
    const dataEnd = Math.min(dataStart + size, u8.length);

    if (fourcc === 'EXIF' || fourcc === 'Exif') {
      // The payload may or may not include the "Exif\0\0" prefix.
      let ts = dataStart;
      if (dataEnd - ts >= 6 &&
          u8[ts] === 0x45 && u8[ts + 1] === 0x78 && u8[ts + 2] === 0x69 &&
          u8[ts + 3] === 0x66 && u8[ts + 4] === 0x00 && u8[ts + 5] === 0x00) {
        ts += 6;
      }
      parseExifTiff(u8, ts, dataEnd, signals, 'webp');
    } else if (fourcc === 'XMP ' || fourcc === 'XMP\0') {
      parseXmp(decodeUtf8(u8, dataStart, dataEnd), signals, 'webp');
    }

    // RIFF chunks are even-aligned.
    let next = dataStart + size;
    if (size % 2 !== 0) next++;
    if (next <= p) break;
    p = next;
  }
}

/* ------------------------------------------------------------------ *
 * Public API
 * ------------------------------------------------------------------ */

/**
 * Scan raw image bytes for metadata provenance signals.
 *
 * @param {Uint8Array} u8 - Raw image bytes.
 * @returns {{ signals: Array<{id:string, kind:string, label:string, detail:string}> }}
 */
function scanMetadata(u8) {
  const signals = [];
  if (!u8 || u8.length < 8) return { signals };

  const bytes = u8 instanceof Uint8Array ? u8 : new Uint8Array(u8);

  try {
    if (isJpeg(bytes)) {
      scanJpeg(bytes, signals);
    } else if (isPng(bytes)) {
      scanPng(bytes, signals);
    } else if (isWebp(bytes)) {
      scanWebp(bytes, signals);
    } else if (isTiff(bytes)) {
      parseExifTiff(bytes, 0, bytes.length, signals, 'tiff');
    }
    // Unrecognised formats produce no signals — absence proves nothing.
  } catch (e) {
    signals.push(sig('meta-error', 'info', 'Metadata parse error',
      trim(String((e && e.message) || e), 120)));
  }

  // De-duplicate identical (id, detail) pairs that can arise from multiple
  // IFDs or overlapping chunks.
  const seen = new Set();
  const deduped = [];
  for (const s of signals) {
    const key = s.id + '\u0000' + s.detail;
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(s);
  }
  return { signals: deduped };
}

/**
 * Fuse metadata signals with a prior calibrated probability (pCal) from the
 * visual model.
 *
 * Rules:
 *   - HARD AI proof (embedded prompt/workflow, C2PA AI assertion, IPTC/XMP
 *     trainedAlgorithmicMedia)  -> max(pCal, 0.97). Never lowers the score.
 *   - STRONG AI indicator (Software/CreatorTool naming a generator, bare C2PA
 *     manifest) -> max(pCal, 0.85).
 *   - 'real' evidence (camera EXIF with MakerNote + DateTimeOriginal) is
 *     context only and does NOT change the score.
 *
 * @param {number} pCal - Prior calibrated P(AI) from the visual model.
 * @param {Array} signals - Output of scanMetadata().
 * @returns {number} Fused P(AI) in [0, 1].
 */
function fuseSignals(pCal, signals) {
  if (typeof pCal !== 'number' || isNaN(pCal)) pCal = 0.5;
  if (!signals || signals.length === 0) return pCal;

  let hardHit = false;
  let strongHit = false;

  for (let i = 0; i < signals.length; i++) {
    const s = signals[i];
    if (s.kind !== 'ai') continue;        // 'real' and 'info' are non-score-changing
    if (HARD_AI_IDS.has(s.id)) {
      hardHit = true;
    } else if (STRONG_AI_IDS.has(s.id)) {
      strongHit = true;
    } else {
      // Any unrecognised 'ai' signal is treated as a strong indicator.
      strongHit = true;
    }
  }

  let score = pCal;
  if (hardHit) {
    score = Math.max(score, 0.97);
  } else if (strongHit) {
    score = Math.max(score, 0.85);
  }
  // Clamp to a sane range.
  if (score < 0) score = 0;
  if (score > 1) score = 1;
  return score;
}

/* ------------------------------------------------------------------ *
 * Exports
 * ------------------------------------------------------------------ */

export {
  scanMetadata,
  fuseSignals,
  AI_GENERATOR_MARKERS,
  IPTC_AI_SOURCETYPES,
  sig,
  trim
};
