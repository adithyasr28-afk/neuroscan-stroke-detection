"""
app.py — Desktop UI for Brain Stroke Detection
Run with: python app.py

Requires your existing files in the same folder:
  preprocess.py, segmentation.py, features.py, detection.py
And a trained model: stroke_model.joblib
"""

'''import tkinter as tk
from tkinter import filedialog, ttk
import threading
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont

# ── Import your existing pipeline ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess_image
from segmentation import skull_strip, segment_stroke
from features import extract_features
from detection import detect_stroke


# ── Colour palette ─────────────────────────────────────────────────────────
BG          = "#0d1117"   # near-black background
PANEL       = "#161b22"   # slightly lighter panels
BORDER      = "#21262d"   # subtle borders
ACCENT      = "#58a6ff"   # blue accent
STROKE_CLR  = "#f85149"   # red  — stroke detected
NORMAL_CLR  = "#3fb950"   # green — normal
TEXT        = "#e6edf3"   # primary text
TEXT_DIM    = "#8b949e"   # secondary text
FONT_MAIN   = ("Segoe UI", 10)
FONT_BOLD   = ("Segoe UI", 10, "bold")
FONT_TITLE  = ("Segoe UI", 13, "bold")
FONT_RESULT = ("Segoe UI", 22, "bold")
FONT_MONO   = ("Courier New", 9)

PREVIEW_W, PREVIEW_H = 340, 280   # image panel dimensions


# ───────────────────────────────────────────────────────────────────────────
class StrokeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Brain Stroke Detection")
        self.configure(bg=BG)
        self.resizable(False, False)

        self._image_path  = None
        self._orig_pil    = None   # original image (PIL)
        self._result      = None   # last detection result dict

        self._build_ui()
        self._center_window()

    # ── Layout ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header bar ────────────────────────────────────────────────────
        header = tk.Frame(self, bg=PANEL, bd=0)
        header.pack(fill="x")

        tk.Label(
            header, text="🧠  Brain Stroke Detection",
            font=FONT_TITLE, bg=PANEL, fg=TEXT, pady=12, padx=20
        ).pack(side="left")

        model_status = self._check_model()
        tk.Label(
            header, text=model_status["label"],
            font=FONT_MAIN, bg=PANEL, fg=model_status["color"],
            padx=20
        ).pack(side="right")

        _divider(self)

        # ── Main body (two columns) ────────────────────────────────────────
        body = tk.Frame(self, bg=BG)
        body.pack(padx=20, pady=16, fill="both")

        # Left column — image panels
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", padx=(0, 16))

        self._canvas_orig = _image_panel(left, "Original Scan", PREVIEW_W, PREVIEW_H)
        tk.Frame(left, height=12, bg=BG).pack()
        self._canvas_bbox = _image_panel(left, "Detected Regions", PREVIEW_W, PREVIEW_H)

        # Right column — controls + results
        right = tk.Frame(body, bg=BG, width=260)
        right.pack(side="left", fill="both", expand=True)
        right.pack_propagate(False)

        # Upload button
        self._btn_upload = _button(
            right, "📂  Upload MRI Image", self._upload_image,
            bg=ACCENT, fg="#0d1117"
        )
        self._btn_upload.pack(fill="x", pady=(0, 8))

        # Analyse button (disabled until image loaded)
        self._btn_analyse = _button(
            right, "🔍  Analyse", self._run_analysis,
            bg=PANEL, fg=TEXT_DIM, state="disabled"
        )
        self._btn_analyse.pack(fill="x", pady=(0, 16))

        _divider(right, horizontal=True)

        # File info
        self._lbl_file = tk.Label(
            right, text="No image loaded",
            font=FONT_MAIN, bg=BG, fg=TEXT_DIM,
            wraplength=240, justify="left"
        )
        self._lbl_file.pack(anchor="w", pady=(10, 0))

        # ── Result card ───────────────────────────────────────────────────
        self._result_frame = tk.Frame(right, bg=PANEL, bd=0)
        self._result_frame.pack(fill="x", pady=14)

        self._lbl_verdict = tk.Label(
            self._result_frame, text="—",
            font=FONT_RESULT, bg=PANEL, fg=TEXT_DIM, pady=6
        )
        self._lbl_verdict.pack()

        self._lbl_conf_title = tk.Label(
            self._result_frame, text="Confidence",
            font=FONT_MAIN, bg=PANEL, fg=TEXT_DIM
        )
        self._lbl_conf_title.pack()

        # Confidence bar
        bar_frame = tk.Frame(self._result_frame, bg=PANEL)
        bar_frame.pack(fill="x", padx=16, pady=(4, 2))

        self._bar_bg = tk.Frame(bar_frame, bg=BORDER, height=10)
        self._bar_bg.pack(fill="x")
        self._bar_fill = tk.Frame(self._bar_bg, bg=TEXT_DIM, height=10, width=0)
        self._bar_fill.place(x=0, y=0, relheight=1)

        self._lbl_conf_val = tk.Label(
            self._result_frame, text="—",
            font=FONT_BOLD, bg=PANEL, fg=TEXT_DIM, pady=4
        )
        self._lbl_conf_val.pack()

        # ── Metrics grid ──────────────────────────────────────────────────
        _divider(right, horizontal=True)

        metrics_title = tk.Label(
            right, text="SCAN DETAILS",
            font=("Segoe UI", 8, "bold"), bg=BG, fg=TEXT_DIM
        )
        metrics_title.pack(anchor="w", pady=(8, 4))

        self._metrics_frame = tk.Frame(right, bg=BG)
        self._metrics_frame.pack(fill="x")

        self._metric_vars = {}
        for key in ["Method", "Asymmetry", "Regions", "Max Area"]:
            row = tk.Frame(self._metrics_frame, bg=BG)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=key, font=FONT_MAIN, bg=BG,
                     fg=TEXT_DIM, width=12, anchor="w").pack(side="left")
            var = tk.StringVar(value="—")
            tk.Label(row, textvariable=var, font=FONT_MONO,
                     bg=BG, fg=TEXT, anchor="w").pack(side="left")
            self._metric_vars[key] = var

        # ── Status bar ────────────────────────────────────────────────────
        _divider(self)
        self._status_var = tk.StringVar(value="Ready — upload an MRI scan to begin.")
        tk.Label(
            self, textvariable=self._status_var,
            font=FONT_MAIN, bg=PANEL, fg=TEXT_DIM,
            anchor="w", padx=16, pady=6
        ).pack(fill="x")

    # ── Actions ────────────────────────────────────────────────────────────
    def _upload_image(self):
        path = filedialog.askopenfilename(
            title="Select Brain MRI/CT Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return

        self._image_path = path
        self._result = None

        # Show original image
        pil = _load_pil(path, PREVIEW_W, PREVIEW_H)
        if pil is None:
            self._status("❌  Could not load image.", TEXT_DIM)
            return

        self._orig_pil = pil
        _show_on_canvas(self._canvas_orig, pil)
        _show_on_canvas(self._canvas_bbox, None)   # clear bbox panel

        # Reset results
        self._lbl_verdict.config(text="—", fg=TEXT_DIM)
        self._lbl_conf_val.config(text="—")
        self._bar_fill.config(width=0, bg=TEXT_DIM)
        for v in self._metric_vars.values():
            v.set("—")

        self._lbl_file.config(text=f"📄  {os.path.basename(path)}", fg=TEXT)
        self._btn_analyse.config(state="normal", bg=ACCENT, fg="#0d1117")
        self._status(f"Image loaded: {os.path.basename(path)}")

    def _run_analysis(self):
        if not self._image_path:
            return

        self._btn_analyse.config(state="disabled", text="Analysing…")
        self._status("Running analysis…")
        self.update()

        # Run in background thread so UI stays responsive
        threading.Thread(target=self._analyse_worker, daemon=True).start()

    def _analyse_worker(self):
        try:
            raw = cv2.imread(self._image_path)
            if raw is None:
                self.after(0, lambda: self._status("❌  Failed to read image.", STROKE_CLR))
                return

            # Full pipeline
            processed = preprocess_image(raw)
            brain     = skull_strip(processed)
            mask      = segment_stroke(brain)
            feats     = extract_features(raw, is_path=False)
            result    = detect_stroke(feats)

            # Draw bounding boxes on a copy of the original
            bbox_img  = self._draw_boxes(raw, mask, result)

            self.after(0, lambda: self._display_result(result, feats, bbox_img))

        except Exception as e:
            self.after(0, lambda: self._status(f"❌  Error: {e}", STROKE_CLR))
        finally:
            self.after(0, lambda: self._btn_analyse.config(
                state="normal", text="🔍  Analyse"))

    def _display_result(self, result, feats, bbox_bgr):
        is_stroke = result["stroke"]
        conf      = result["confidence"]
        method    = result["method"]

        # Verdict label
        verdict_text  = "STROKE DETECTED" if is_stroke else "NORMAL"
        verdict_color = STROKE_CLR if is_stroke else NORMAL_CLR
        self._lbl_verdict.config(text=verdict_text, fg=verdict_color)

        # Confidence bar
        bar_w = int((self._bar_bg.winfo_width() or 220) * conf)
        self._bar_fill.config(width=max(bar_w, 2), bg=verdict_color)
        self._lbl_conf_val.config(
            text=f"{conf:.1%}", fg=verdict_color
        )

        # Metrics
        # Asymmetry and region features sit at fixed offsets in the vector
        asymmetry  = feats[-8]
        num_regions = int(feats[-6])
        max_area   = int(feats[-4])

        self._metric_vars["Method"].set(method.upper())
        self._metric_vars["Asymmetry"].set(f"{asymmetry:.2f}")
        self._metric_vars["Regions"].set(str(num_regions))
        self._metric_vars["Max Area"].set(str(max_area))

        # Bbox image
        bbox_pil = _bgr_to_pil(bbox_bgr, PREVIEW_W, PREVIEW_H)
        _show_on_canvas(self._canvas_bbox, bbox_pil)

        self._status(
            f"{'⚠️  Stroke detected' if is_stroke else '✅  No stroke detected'}"
            f"  |  Confidence: {conf:.1%}  |  Method: {method.upper()}"
        )

    # ── Drawing ────────────────────────────────────────────────────────────
    def _draw_boxes(self, raw_bgr, mask, result):
        out = raw_bgr.copy()
        color = (88, 49, 248) if result["stroke"] else (80, 185, 63)

        orig_h, orig_w = raw_bgr.shape[:2]
        mask_h, mask_w = mask.shape[:2]

        # Scale factors from 128x128 mask → original image size
        scale_x = orig_w / mask_w
        scale_y = orig_h / mask_h

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if cv2.contourArea(cnt) < 80:
                continue
            x, y, w, h = cv2.boundingRect(cnt)

            # Scale coordinates up to original image dimensions
            x1 = int(x * scale_x)
            y1 = int(y * scale_y)
            x2 = int((x + w) * scale_x)
            y2 = int((y + h) * scale_y)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label overlay
        label = f"{'STROKE' if result['stroke'] else 'NORMAL'}  {result['confidence']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (8, 8), (14 + tw, 18 + th), (13, 17, 23), -1)
        cv2.putText(out, label, (11, 11 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 1, cv2.LINE_AA)
        return out

    # ── Helpers ────────────────────────────────────────────────────────────
    def _status(self, msg, color=TEXT_DIM):
        self._status_var.set(msg)

    def _check_model(self):
        model_file = os.path.join(os.path.dirname(__file__), "stroke_model.joblib")
        if os.path.exists(model_file):
            return {"label": "● Model loaded", "color": NORMAL_CLR}
        return {"label": "○ No model — heuristic mode", "color": "#d29922"}

    def _center_window(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")


# ── Widget helpers ──────────────────────────────────────────────────────────

def _divider(parent, horizontal=False):
    if horizontal:
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=4)
    else:
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x")


def _button(parent, text, command, bg=PANEL, fg=TEXT, state="normal"):
    btn = tk.Label(
        parent, text=text,
        font=FONT_BOLD, bg=bg, fg=fg,
        pady=9, cursor="hand2",
        relief="flat"
    )
    btn.bind("<Button-1>", lambda e: command() if btn["state"] != "disabled" else None)
    btn.bind("<Enter>", lambda e: btn.config(bg=_lighten(bg)) if btn["state"] != "disabled" else None)
    btn.bind("<Leave>", lambda e: btn.config(bg=bg))
    btn._orig_bg = bg
    btn._state   = state

    def set_state(s):
        btn._state = s
        if s == "disabled":
            btn.config(fg=TEXT_DIM, bg=PANEL, cursor="arrow")
        else:
            btn.config(fg=fg, bg=bg, cursor="hand2")
    btn.config  = btn.configure
    btn._set_state = set_state

    # Patch config to intercept state= kwarg
    _orig_config = btn.configure
    def patched_config(**kw):
        if "state" in kw:
            set_state(kw.pop("state"))
        if kw:
            _orig_config(**kw)
    btn.config = patched_config

    return btn


def _image_panel(parent, title, w, h):
    """Labelled frame containing a canvas for image display."""
    frame = tk.Frame(parent, bg=PANEL, bd=0)
    frame.pack()

    tk.Label(frame, text=title, font=FONT_MAIN,
             bg=PANEL, fg=TEXT_DIM, pady=4).pack()

    canvas = tk.Canvas(frame, width=w, height=h,
                       bg=BG, highlightthickness=1,
                       highlightbackground=BORDER)
    canvas.pack(padx=1, pady=(0, 6))

    # Placeholder text
    canvas.create_text(w // 2, h // 2, text="No image",
                       fill=TEXT_DIM, font=FONT_MAIN, tags="placeholder")
    return canvas


def _show_on_canvas(canvas, pil_img):
    """Render a PIL image (or None) on a canvas."""
    canvas.delete("all")
    if pil_img is None:
        w = int(canvas["width"])
        h = int(canvas["height"])
        canvas.create_text(w // 2, h // 2, text="No image",
                           fill=TEXT_DIM, font=FONT_MAIN)
        return
    tk_img = ImageTk.PhotoImage(pil_img)
    canvas._tk_img = tk_img   # keep reference
    cw, ch = int(canvas["width"]), int(canvas["height"])
    canvas.create_image(cw // 2, ch // 2, anchor="center", image=tk_img)


def _load_pil(path, max_w, max_h):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        return img
    except Exception:
        return None


def _bgr_to_pil(bgr, max_w, max_h):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil.thumbnail((max_w, max_h), Image.LANCZOS)
    return pil


def _lighten(hex_color):
    """Return a slightly lighter version of a hex colour."""
    try:
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        r = min(255, r + 20)
        g = min(255, g + 20)
        b = min(255, b + 20)
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return hex_color


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = StrokeApp()
    app.mainloop()
    '''
"""
app.py — Desktop UI for Brain Stroke Detection (with subtype classification)
Run with: python app.py
"""

import tkinter as tk
from tkinter import filedialog
import threading
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import preprocess_image
from segmentation import skull_strip, segment_stroke
from features import extract_features
from detection import detect_stroke


# ── Colour palette ────────────────────────────────────────────────────────
BG           = "#0d1117"
PANEL        = "#161b22"
BORDER       = "#21262d"
ACCENT       = "#58a6ff"
STROKE_CLR   = "#f85149"   # red   — stroke detected
NORMAL_CLR   = "#3fb950"   # green — normal
ISCHEMIC_CLR = "#e3b341"   # amber — ischemic
HEMO_CLR     = "#f85149"   # red   — hemorrhagic
MIXED_CLR    = "#d2a8ff"   # purple — mixed
TEXT         = "#e6edf3"
TEXT_DIM     = "#8b949e"
FONT_MAIN    = ("Segoe UI", 10)
FONT_BOLD    = ("Segoe UI", 10, "bold")
FONT_TITLE   = ("Segoe UI", 13, "bold")
FONT_RESULT  = ("Segoe UI", 20, "bold")
FONT_SMALL   = ("Segoe UI", 9)
FONT_MONO    = ("Courier New", 9)

PREVIEW_W, PREVIEW_H = 320, 260

# BGR colours for OpenCV drawing
BOX_ISCHEMIC  = (0,   179, 227)   # amber-ish in BGR
BOX_HEMO      = (80,  81,  248)   # red in BGR
BOX_MIXED     = (255, 168, 210)   # purple in BGR
BOX_NORMAL    = (80,  185,  63)   # green in BGR


# ─────────────────────────────────────────────────────────────────────────
class StrokeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Brain Stroke Detection")
        self.configure(bg=BG)
        self.resizable(False, False)
        self._image_path = None
        self._build_ui()
        self._center_window()

    # ── Layout ────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        header = tk.Frame(self, bg=PANEL)
        header.pack(fill="x")
        tk.Label(header, text="🧠  Brain Stroke Detection",
                 font=FONT_TITLE, bg=PANEL, fg=TEXT,
                 pady=12, padx=20).pack(side="left")

        model_info = self._check_model()
        tk.Label(header, text=model_info["label"],
                 font=FONT_MAIN, bg=PANEL,
                 fg=model_info["color"], padx=20).pack(side="right")

        _divider(self)

        # Body
        body = tk.Frame(self, bg=BG)
        body.pack(padx=18, pady=14, fill="both")

        # ── Left: image panels ─────────────────────────────────────────
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", padx=(0, 14))

        self._canvas_orig = _image_panel(left, "Original Scan",    PREVIEW_W, PREVIEW_H)
        tk.Frame(left, height=10, bg=BG).pack()
        self._canvas_bbox = _image_panel(left, "Detected Regions", PREVIEW_W, PREVIEW_H)

        # Legend below images
        legend = tk.Frame(left, bg=BG)
        legend.pack(pady=(6, 0))
        for colour, label in [
            (ISCHEMIC_CLR, "Ischemic"),
            (HEMO_CLR,     "Hemorrhagic"),
            (MIXED_CLR,    "Mixed"),
            (NORMAL_CLR,   "Normal"),
        ]:
            dot = tk.Label(legend, text="●", font=FONT_SMALL, bg=BG, fg=colour)
            dot.pack(side="left", padx=(0, 2))
            tk.Label(legend, text=label, font=FONT_SMALL,
                     bg=BG, fg=TEXT_DIM).pack(side="left", padx=(0, 10))

        # ── Right: controls + results ──────────────────────────────────
        right = tk.Frame(body, bg=BG, width=270)
        right.pack(side="left", fill="both", expand=True)
        right.pack_propagate(False)

        self._btn_upload = _button(right, "📂  Upload MRI Image",
                                   self._upload_image, bg=ACCENT, fg="#0d1117")
        self._btn_upload.pack(fill="x", pady=(0, 8))

        self._btn_analyse = _button(right, "🔍  Analyse",
                                    self._run_analysis, bg=PANEL,
                                    fg=TEXT_DIM, state="disabled")
        self._btn_analyse.pack(fill="x", pady=(0, 14))

        _divider(right, horizontal=True)

        self._lbl_file = tk.Label(right, text="No image loaded",
                                  font=FONT_MAIN, bg=BG, fg=TEXT_DIM,
                                  wraplength=250, justify="left")
        self._lbl_file.pack(anchor="w", pady=(8, 0))

        # ── Binary result card ────────────────────────────────────────
        self._result_frame = tk.Frame(right, bg=PANEL)
        self._result_frame.pack(fill="x", pady=12)

        self._lbl_verdict = tk.Label(self._result_frame, text="—",
                                     font=FONT_RESULT, bg=PANEL,
                                     fg=TEXT_DIM, pady=6)
        self._lbl_verdict.pack()

        tk.Label(self._result_frame, text="Confidence",
                 font=FONT_MAIN, bg=PANEL, fg=TEXT_DIM).pack()

        bar_frame = tk.Frame(self._result_frame, bg=PANEL)
        bar_frame.pack(fill="x", padx=16, pady=(4, 2))
        self._bar_bg = tk.Frame(bar_frame, bg=BORDER, height=10)
        self._bar_bg.pack(fill="x")
        self._bar_fill = tk.Frame(self._bar_bg, bg=TEXT_DIM, height=10, width=0)
        self._bar_fill.place(x=0, y=0, relheight=1)

        self._lbl_conf = tk.Label(self._result_frame, text="—",
                                  font=FONT_BOLD, bg=PANEL,
                                  fg=TEXT_DIM, pady=2)
        self._lbl_conf.pack()

        # ── Subtype card ──────────────────────────────────────────────
        self._type_frame = tk.Frame(right, bg=PANEL)
        self._type_frame.pack(fill="x", pady=(0, 12))

        tk.Label(self._type_frame, text="STROKE TYPE",
                 font=("Segoe UI", 8, "bold"), bg=PANEL,
                 fg=TEXT_DIM, pady=4).pack()

        self._lbl_type = tk.Label(self._type_frame, text="—",
                                  font=("Segoe UI", 16, "bold"),
                                  bg=PANEL, fg=TEXT_DIM, pady=2)
        self._lbl_type.pack()

        self._lbl_type_desc = tk.Label(
            self._type_frame, text="",
            font=FONT_SMALL, bg=PANEL, fg=TEXT_DIM,
            wraplength=250, justify="center", pady=4
        )
        self._lbl_type_desc.pack()

        self._lbl_type_conf = tk.Label(self._type_frame, text="",
                                       font=FONT_SMALL, bg=PANEL, fg=TEXT_DIM)
        self._lbl_type_conf.pack(pady=(0, 6))

        # ── Scan details ──────────────────────────────────────────────
        _divider(right, horizontal=True)
        tk.Label(right, text="SCAN DETAILS",
                 font=("Segoe UI", 8, "bold"), bg=BG,
                 fg=TEXT_DIM).pack(anchor="w", pady=(8, 4))

        self._metric_frame = tk.Frame(right, bg=BG)
        self._metric_frame.pack(fill="x")
        self._metric_vars = {}
        for key in ["Method", "Asymmetry", "Dark Area", "Bright Area"]:
            row = tk.Frame(self._metric_frame, bg=BG)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=key, font=FONT_MAIN, bg=BG,
                     fg=TEXT_DIM, width=12, anchor="w").pack(side="left")
            var = tk.StringVar(value="—")
            tk.Label(row, textvariable=var, font=FONT_MONO,
                     bg=BG, fg=TEXT, anchor="w").pack(side="left")
            self._metric_vars[key] = var

        # Status bar
        _divider(self)
        self._status_var = tk.StringVar(value="Ready — upload an MRI scan to begin.")
        tk.Label(self, textvariable=self._status_var,
                 font=FONT_MAIN, bg=PANEL, fg=TEXT_DIM,
                 anchor="w", padx=16, pady=6).pack(fill="x")

    # ── Actions ───────────────────────────────────────────────────────────
    def _upload_image(self):
        path = filedialog.askopenfilename(
            title="Select Brain MRI/CT Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                       ("All files", "*.*")]
        )
        if not path:
            return

        self._image_path = path
        pil = _load_pil(path, PREVIEW_W, PREVIEW_H)
        if pil is None:
            self._status("❌  Could not load image.")
            return

        _show_on_canvas(self._canvas_orig, pil)
        _show_on_canvas(self._canvas_bbox, None)
        self._reset_results()
        self._lbl_file.config(text=f"📄  {os.path.basename(path)}", fg=TEXT)
        self._btn_analyse.config(state="normal", bg=ACCENT, fg="#0d1117")
        self._status(f"Image loaded: {os.path.basename(path)}")

    def _run_analysis(self):
        if not self._image_path:
            return
        self._btn_analyse.config(state="disabled", text="Analysing…")
        self._status("Running analysis…")
        self.update()
        threading.Thread(target=self._analyse_worker, daemon=True).start()

    def _analyse_worker(self):
        try:
            raw = cv2.imread(self._image_path)
            if raw is None:
                self.after(0, lambda: self._status("❌  Failed to read image."))
                return

            processed                        = preprocess_image(raw)
            brain                            = skull_strip(processed)
            combined, dark_mask, bright_mask = segment_stroke(brain)
            feats                            = extract_features(raw, is_path=False)
            result                           = detect_stroke(feats, dark_mask, bright_mask)
            bbox_img                         = self._draw_boxes(
                                                   raw, combined, dark_mask,
                                                   bright_mask, result)

            self.after(0, lambda: self._display_result(result, feats, bbox_img))

        except Exception as e:
            self.after(0, lambda: self._status(f"❌  Error: {e}"))
        finally:
            self.after(0, lambda: self._btn_analyse.config(
                state="normal", text="🔍  Analyse"))

    def _display_result(self, result, feats, bbox_bgr):
        is_stroke   = result["stroke"]
        conf        = result["confidence"]
        stroke_type = result["stroke_type"]
        method      = result["method"]

        # Binary verdict
        verdict_color = STROKE_CLR if is_stroke else NORMAL_CLR
        verdict_text  = "STROKE DETECTED" if is_stroke else "NORMAL"
        self._lbl_verdict.config(text=verdict_text, fg=verdict_color)

        bar_w = int((self._bar_bg.winfo_width() or 220) * conf)
        self._bar_fill.config(width=max(bar_w, 2), bg=verdict_color)
        self._lbl_conf.config(text=f"{conf:.1%}", fg=verdict_color)

        # Subtype
        type_color, type_desc = self._type_info(stroke_type)
        self._lbl_type.config(text=stroke_type.upper(), fg=type_color)
        self._lbl_type_desc.config(text=type_desc, fg=TEXT_DIM)
        self._lbl_type_conf.config(
            text=f"Subtype confidence: {result['type_confidence']}", fg=TEXT_DIM)

        # Metrics
        asymmetry = feats[-14] if len(feats) > 14 else 0
        self._metric_vars["Method"].set(method.upper())
        self._metric_vars["Asymmetry"].set(f"{asymmetry:.2f}")
        self._metric_vars["Dark Area"].set(str(result["dark_area"]))
        self._metric_vars["Bright Area"].set(str(result["bright_area"]))

        bbox_pil = _bgr_to_pil(bbox_bgr, PREVIEW_W, PREVIEW_H)
        _show_on_canvas(self._canvas_bbox, bbox_pil)

        status_icon = "⚠️" if is_stroke else "✅"
        self._status(
            f"{status_icon}  {verdict_text}  |  Type: {stroke_type.upper()}"
            f"  |  Conf: {conf:.1%}  |  {method.upper()}"
        )

    # ── Drawing ───────────────────────────────────────────────────────────
    def _draw_boxes(self, raw_bgr, combined, dark_mask, bright_mask, result):
        """
        Draw colour-coded bounding boxes:
          Amber  → ischemic regions   (dark areas)
          Red    → hemorrhagic regions (bright areas)
          Purple → overlap / mixed
          Green  → normal (no stroke)
        """
        out = raw_bgr.copy()
        orig_h, orig_w = raw_bgr.shape[:2]
        mask_h, mask_w = combined.shape[:2]
        sx = orig_w / mask_w
        sy = orig_h / mask_h

        def draw_contours(mask, color):
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 80:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                x1, y1 = int(x * sx), int(y * sy)
                x2, y2 = int((x + w) * sx), int((y + h) * sy)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        if result["stroke"]:
            draw_contours(dark_mask,   BOX_ISCHEMIC)
            draw_contours(bright_mask, BOX_HEMO)
        else:
            draw_contours(combined, BOX_NORMAL)

        # Label overlay
        stroke_type = result["stroke_type"]
        label = f"{stroke_type.upper()}  {result['confidence']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (8, 8), (14 + tw, 16 + th), (13, 17, 23), -1)
        cv2.putText(out, label, (11, 11 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    BOX_ISCHEMIC if stroke_type == "ischemic"
                    else BOX_HEMO if stroke_type == "hemorrhagic"
                    else BOX_MIXED if stroke_type == "mixed"
                    else BOX_NORMAL,
                    1, cv2.LINE_AA)
        return out

    # ── Helpers ───────────────────────────────────────────────────────────
    def _type_info(self, stroke_type):
        """Return (colour, description) for a given stroke type."""
        info = {
            "ischemic":     (ISCHEMIC_CLR,
                             "Blocked blood flow — dark region on scan.\n"
                             "Requires clot-busting treatment."),
            "hemorrhagic":  (HEMO_CLR,
                             "Bleeding in the brain — bright region on scan.\n"
                             "Requires pressure management."),
            "mixed":        (MIXED_CLR,
                             "Both ischemic and hemorrhagic features detected.\n"
                             "Further clinical review recommended."),
            "normal":       (NORMAL_CLR,
                             "No significant lesion regions detected."),
        }
        return info.get(stroke_type, (TEXT_DIM, ""))

    def _reset_results(self):
        self._lbl_verdict.config(text="—", fg=TEXT_DIM)
        self._lbl_conf.config(text="—", fg=TEXT_DIM)
        self._bar_fill.config(width=0, bg=TEXT_DIM)
        self._lbl_type.config(text="—", fg=TEXT_DIM)
        self._lbl_type_desc.config(text="")
        self._lbl_type_conf.config(text="")
        for v in self._metric_vars.values():
            v.set("—")

    def _status(self, msg):
        self._status_var.set(msg)

    def _check_model(self):
        path = os.path.join(os.path.dirname(__file__), "stroke_model.joblib")
        if os.path.exists(path):
            return {"label": "● Model loaded", "color": NORMAL_CLR}
        return {"label": "○ No model — heuristic mode", "color": "#d29922"}

    def _center_window(self):
        self.update_idletasks()
        w  = self.winfo_width()
        h  = self.winfo_height()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"+{(sw - w) // 2}+{(sh - h) // 2}")


# ── Widget helpers ─────────────────────────────────────────────────────────

def _divider(parent, horizontal=False):
    tk.Frame(parent, bg=BORDER, height=1).pack(
        fill="x", pady=4 if horizontal else 0)


def _button(parent, text, command, bg=PANEL, fg=TEXT, state="normal"):
    btn = tk.Label(parent, text=text, font=FONT_BOLD,
                   bg=bg, fg=fg, pady=9, cursor="hand2", relief="flat")
    orig_bg = bg

    def on_click(e):
        if getattr(btn, "_disabled", False):
            return
        command()

    def on_enter(e):
        if not getattr(btn, "_disabled", False):
            btn.config(bg=_lighten(orig_bg))

    def on_leave(e):
        if not getattr(btn, "_disabled", False):
            btn.config(bg=orig_bg)

    btn.bind("<Button-1>", on_click)
    btn.bind("<Enter>",    on_enter)
    btn.bind("<Leave>",    on_leave)
    btn._disabled = (state == "disabled")

    _orig_config = btn.configure

    def patched_config(**kw):
        if "state" in kw:
            s = kw.pop("state")
            btn._disabled = (s == "disabled")
            btn.config(fg=TEXT_DIM if btn._disabled else fg,
                       cursor="arrow" if btn._disabled else "hand2")
        if "text" in kw:
            btn.configure(text=kw.pop("text"))
        if kw:
            _orig_config(**kw)

    btn.config = patched_config
    return btn


def _image_panel(parent, title, w, h):
    frame = tk.Frame(parent, bg=PANEL)
    frame.pack()
    tk.Label(frame, text=title, font=FONT_MAIN,
             bg=PANEL, fg=TEXT_DIM, pady=4).pack()
    canvas = tk.Canvas(frame, width=w, height=h, bg=BG,
                       highlightthickness=1, highlightbackground=BORDER)
    canvas.pack(padx=1, pady=(0, 4))
    canvas.create_text(w // 2, h // 2, text="No image",
                       fill=TEXT_DIM, font=FONT_MAIN)
    return canvas


def _show_on_canvas(canvas, pil_img):
    canvas.delete("all")
    if pil_img is None:
        w = int(canvas["width"])
        h = int(canvas["height"])
        canvas.create_text(w // 2, h // 2, text="No image",
                           fill=TEXT_DIM, font=FONT_MAIN)
        return
    tk_img = ImageTk.PhotoImage(pil_img)
    canvas._tk_img = tk_img
    cw = int(canvas["width"])
    ch = int(canvas["height"])
    canvas.create_image(cw // 2, ch // 2, anchor="center", image=tk_img)


def _load_pil(path, max_w, max_h):
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        return img
    except Exception:
        return None


def _bgr_to_pil(bgr, max_w, max_h):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    pil.thumbnail((max_w, max_h), Image.LANCZOS)
    return pil


def _lighten(hex_color):
    try:
        r = min(255, int(hex_color[1:3], 16) + 20)
        g = min(255, int(hex_color[3:5], 16) + 20)
        b = min(255, int(hex_color[5:7], 16) + 20)
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return hex_color


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = StrokeApp()
    app.mainloop()