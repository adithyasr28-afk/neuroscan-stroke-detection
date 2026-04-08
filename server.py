"""
server.py — Flask web server for NeuroScan stroke detection.

Run:
    python server.py

Then open:
    http://localhost:5000

Place this file in the same folder as your other pipeline files:
    preprocess.py, segmentation.py, features.py, detection.py, model.py
"""

import os
import sys
import base64
import traceback

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template

# ── Make sure sibling modules are importable ──────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from preprocess  import preprocess_image
from segmentation import skull_strip, segment_stroke
from features    import extract_features
from detection   import detect_stroke

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024   # 32 MB max upload

ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


# ── Helper: draw annotated bounding boxes ─────────────────────────────────

def _draw_boxes(raw_bgr, combined, dark_mask, bright_mask, result):
    """
    Draw colour-coded boxes on the original image.
      Amber (#4a9eff-ish in BGR) → ischemic (dark regions)
      Red                        → hemorrhagic (bright regions)
      Green                      → normal
    """
    out = raw_bgr.copy()
    orig_h, orig_w = raw_bgr.shape[:2]
    mask_h, mask_w = combined.shape[:2]
    sx = orig_w / mask_w
    sy = orig_h / mask_h

    # BGR colours
    COL_ISCHEMIC  = (255, 158, 74)    # amber
    COL_HEMO      = (80,  81,  248)   # red
    COL_NORMAL    = (80,  185,  63)   # green

    def draw_mask(mask, color):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 80:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            x1, y1 = int(x * sx), int(y * sy)
            x2, y2 = int((x + w) * sx), int((y + h) * sy)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

    if result['stroke']:
        draw_mask(dark_mask,   COL_ISCHEMIC)
        draw_mask(bright_mask, COL_HEMO)
    else:
        draw_mask(combined, COL_NORMAL)

    # Label overlay in top-left
    label = f"{result['stroke_type'].upper()}  {result['confidence']:.0%}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(out, (8, 8), (14 + tw, 18 + th), (13, 17, 23), -1)
    label_col = COL_ISCHEMIC if result['stroke_type'] == 'ischemic' \
           else COL_HEMO     if result['stroke_type'] == 'hemorrhagic' \
           else COL_NORMAL
    cv2.putText(out, label, (11, 11 + th),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_col, 1, cv2.LINE_AA)
    return out


# ── Routes ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/detect', methods=['POST'])
def detect():
    """
    Accepts: multipart/form-data with field 'image'
    Returns: JSON with stroke detection result + base64-encoded annotated image
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    ext  = os.path.splitext(file.filename.lower())[1]

    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'error': f'Unsupported file type: {ext}'}), 400

    try:
        # Decode uploaded bytes → numpy array
        file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
        raw        = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if raw is None:
            return jsonify({'error': 'Could not decode image. Is it a valid brain scan?'}), 400

        # ── Run full pipeline ─────────────────────────────────────────
        processed                        = preprocess_image(raw)
        brain                            = skull_strip(processed)
        combined, dark_mask, bright_mask = segment_stroke(brain)
        feats                            = extract_features(raw, is_path=False)

        if feats is None:
            return jsonify({'error': 'Feature extraction failed'}), 500

        result = detect_stroke(feats, dark_mask, bright_mask)

        # ── Annotated image → base64 ──────────────────────────────────
        annotated     = _draw_boxes(raw, combined, dark_mask, bright_mask, result)
        _, buf         = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 88])
        annotated_b64 = base64.b64encode(buf).decode('utf-8')

        return jsonify({
            'stroke':          result['stroke'],
            'confidence':      round(result['confidence'], 4),
            'method':          result['method'],
            'stroke_type':     result['stroke_type'],
            'dark_area':       result['dark_area'],
            'bright_area':     result['bright_area'],
            'type_confidence': result['type_confidence'],
            'annotated_image': annotated_b64
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/status')
def status():
    """Quick health-check endpoint."""
    model_file = os.path.join(os.path.dirname(__file__), 'stroke_model.joblib')
    return jsonify({
        'status':       'ok',
        'model_loaded': os.path.exists(model_file)
    })


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  NeuroScan — Stroke Detection Server")
    print("="*50)

    model_file = os.path.join(os.path.dirname(__file__), 'stroke_model.joblib')
    if os.path.exists(model_file):
        print("  ✓ Model file found — ML mode active")
    else:
        print("  ⚠  No model file found — heuristic fallback mode")
        print("     Run: python model.py --dataset path/to/dataset")

    print("\n  Open in browser: http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
