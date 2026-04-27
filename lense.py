#!/usr/bin/env python3
import json
import os
import re
import sys
import ast
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Try SciPy smoothing; fallback to numpy moving average if SciPy isn't available
try:
    from scipy.ndimage import uniform_filter1d as _uniform_filter1d
    def smooth_series(y, k):
        k = max(1, int(k))
        return _uniform_filter1d(y, size=k, mode="nearest")
except Exception:
    def smooth_series(y, k):
        """Fast moving average fallback (same length as y)."""
        k = max(1, int(k))
        y = np.asarray(y, dtype=float)
        if k == 1 or len(y) < 2:
            return y.copy()
        kernel = np.ones(k, dtype=float) / k
        pad = k // 2
        ypad = np.pad(y, (pad, k - 1 - pad), mode="edge")
        sm = np.convolve(ypad, kernel, mode="valid")
        return sm

# Optional clipboard support
try:
    import pyperclip  # type: ignore
    _HAS_PYPERCLIP = True
except Exception:
    _HAS_PYPERCLIP = False

# ----------------------------
# Globals filled by the GUI
# ----------------------------
output_labels = []
input_box: ScrolledText | None = None
output_frame: ttk.Frame | None = None

# ----------------------------
# Parsing
# ----------------------------
def parse_loss_input(raw_input: str):
    """
    Robustly parse logs from:
      - JSONL (one dict per line)
      - Console dumps with single quotes
      - Pasted blobs with {...}{...} etc.

    Returns a list[dict] with keys like 'loss', 'epoch', 'learning_rate', ...
    """
    parsed: list[dict] = []

    # First pass: JSONL one-per-line
    for line in raw_input.splitlines():
        line = line.strip()
        if not line:
            continue
        entry = None
        try:
            entry = json.loads(line)
        except Exception:
            # Accept single-quoted Python dicts via ast.literal_eval (safer than regex)
            try:
                entry = ast.literal_eval(line)
            except Exception:
                entry = None
        if isinstance(entry, dict) and "loss" in entry:
            parsed.append(entry)

    if parsed:
        return parsed

    # Fallback: scan for {...} blocks inside the blob
    try:
        objs = re.findall(r"\{[^{}]+\}", raw_input)
        for obj in objs:
            fixed = obj.replace("'", '"')
            entry = json.loads(fixed)
            if isinstance(entry, dict) and "loss" in entry:
                parsed.append(entry)
    except Exception:
        pass

    return parsed



# ----------------------------
# Plotting & metrics
# ----------------------------
def plot_loss_curve(logs: list[dict]):
    global output_labels, output_frame

    # Extract
    steps = list(range(1, len(logs) + 1))
    losses = [e["loss"] for e in logs if "loss" in e]
    epochs = [e.get("epoch", 0.0) for e in logs if "loss" in e]

    if not losses or len(losses) < 3:
        try:
            messagebox.showerror("Error", "Too few valid 'loss' values for analysis.")
        except Exception:
            print("Too few valid 'loss' values for analysis.")
        return

    # Exposure estimate (fraction of an epoch at the tail)
    last_epoch = float(epochs[-1]) if epochs else 0.0
    exposure = last_epoch - int(last_epoch) if last_epoch >= 1.0 else last_epoch
    exposure_pct = round(exposure * 100.0, 2)

    # Arrays
    x = np.array(steps, dtype=float)
    y = np.array(losses, dtype=float)

    # Smoothing
    smoothing_k = 7 if len(y) >= 7 else max(3, (len(y) // 3) * 2 + 1)
    smoothed = smooth_series(y, smoothing_k)

    # ---------------- Holistic stats (full series) ----------------
    y_mean = float(np.mean(y))
    y_med = float(np.median(y))
    y_std = float(np.std(y))
    y_min = float(np.min(y)); idx_min = int(np.argmin(y)) + 1
    y_max = float(np.max(y)); idx_max = int(np.argmax(y)) + 1

    # Variance across session (peak→trough)
    session_drop_abs = float(y_max - y_min)
    session_drop_pct = 100.0 * session_drop_abs / max(y_max, 1e-8)

    # Avg improvement vs start
    start_loss = float(y[0])
    avg_improve_abs = float(start_loss - y_mean)
    avg_improve_pct = 100.0 * avg_improve_abs / max(start_loss, 1e-8)

    # Global slope via linear fit over full smoothed series
    g_coef = np.polyfit(x, smoothed, 1)
    global_slope = float(g_coef[0])          # loss change per step (global)
    global_slope_norm = global_slope * 100.0 # per 100 steps

    # Global noise via MAD
    g_med = float(np.median(y))
    g_mad = float(1.4826 * np.median(np.abs(y - g_med)))
    g_noise = g_mad if g_mad > 1e-8 else float(np.std(y))
    g_snr = float(abs(global_slope) / (g_noise + 1e-8))

    # ---------------- Recent window stats (context) ----------------
    win = min(max(10, len(y) // 10), 200)
    xw = x[-win:]
    yw = smoothed[-win:]
    w = np.linspace(0.5, 1.0, num=len(xw))   # heavier weight on the tail
    r_coef = np.polyfit(xw, yw, 1, w=w)
    slope = float(r_coef[0])                 # per step (recent)
    slope_norm = slope * 100.0               # per 100 steps

    # Recent noise & SNR
    r_med = float(np.median(yw))
    r_mad = float(1.4826 * np.median(np.abs(yw - r_med)))
    noise = r_mad if r_mad > 1e-8 else float(np.std(yw))
    snr = float(abs(slope) / (noise + 1e-8))

    # Best loss & recency
    best_idx = int(np.argmin(y))
    best_loss = float(y[best_idx])
    steps_since_best = len(y) - (best_idx + 1)

    # Predicted stop (label only; no on-plot line)
    eps = 0.05
    target = best_loss + eps
    if slope < -1e-6:
        steps_to_target = max(0, int((y[-1] - target) / -slope))
        predicted_stop_text = f"step {steps[-1] + steps_to_target} (~{steps_to_target} more)"
    else:
        predicted_stop_text = "—"

    # Trend strings (kept)
    def trend_from(norm):
        return (
            "Strong Drop" if norm <= -3.0 else
            "Moderate Drop" if norm <= -1.0 else
            "Weak Drop" if norm < -0.1 else
            "Flat" if abs(norm) <= 0.1 else
            "Weak Rise" if norm < 1.0 else
            "Moderate Rise" if norm < 3.0 else
            "Strong Rise"
        )
    trend_recent = trend_from(slope_norm)
    trend_global = trend_from(global_slope_norm)

    # ------------- UI metrics -------------
    metrics_rows = [
        ("Steps (total)",           f"{len(steps):,}"),
        ("Exposure",                f"{exposure_pct:.2f}% (floor 18%)"),
        ("Loss avg / med",          f"{y_mean:.4f} / {y_med:.4f}"),
        ("Loss std (global)",       f"{y_std:.4f}"),
        ("Best / @step",            f"{best_loss:.4f} @ {best_idx+1}"),
        ("Worst / @step",           f"{y_max:.4f} @ {idx_max}"),
        ("Avg vs start",            f"-{avg_improve_abs:.4f} ({avg_improve_pct:.1f}%)"),
        ("Variance (peak→trough)",  f"-{session_drop_abs:.4f} ({session_drop_pct:.1f}%)"),
        ("Slope (global)",          f"{global_slope_norm:+.3f} per 100 steps"),
        ("SNR (global)",            f"{g_snr:.2f}"),
        ("Slope (recent)",          f"{slope_norm:+.3f} per 100 steps"),
        ("SNR (recent)",            f"{snr:.2f}"),
        ("Steps since best",        f"{steps_since_best}"),
        ("Trend (global/recent)",   f"{trend_global} / {trend_recent}"),
        ("Predicted stop (est.)",   predicted_stop_text),
    ]

    # wipe & render metrics (if GUI present)
    if output_frame is not None:
        for lbl in output_labels:
            try:
                lbl.destroy()
            except Exception:
                pass
        output_labels.clear()
        for idx, (metric_key, metric_val) in enumerate(metrics_rows):
            lbl1 = ttk.Label(output_frame, text=metric_key, anchor="w", font=("Segoe UI", 10))
            lbl1.grid(row=idx, column=0, sticky="w", padx=(4, 2))
            lbl2 = ttk.Label(output_frame, text=metric_val, anchor="w", font=("Segoe UI", 10))
            lbl2.grid(row=idx, column=1, sticky="w", padx=(2, 10))
            output_labels.extend([lbl1, lbl2])

    # ------------- Plot -------------
    fig, ax1 = plt.subplots(figsize=(12, 4))

    # Raw and smoothed loss
    loss_line, = ax1.plot(steps, y, alpha=0.25, label="_nolegend_")
    smooth_line, = ax1.plot(steps, smoothed, label="_nolegend_")

    # Best-loss marker
    ax1.scatter([best_idx + 1], [best_loss], color="green", s=36, zorder=3)

    # Ideal zone band (best .. best+0.05)
    ax1.axhspan(best_loss, best_loss + 0.05, color="green", alpha=0.08)

    # Outlier highlights vs smooth (last window)
    outlier_present = False
    if len(yw) > 1:
        # Use MAD for robust outlier detection (already calculated as r_mad)
        if r_mad > 1e-8:
            for i in range(win):
                idx = len(steps) - win + i
                if 0 < idx < len(smoothed):
                    if abs(y[idx] - smoothed[idx]) > r_mad * 1.5:
                        ax1.axvspan(idx - 0.5, idx + 0.5, color="red", alpha=0.05)
                        outlier_present = True

    # Epoch boundary markers (whole epoch transitions)
    drew_epoch_lines = False
    for i in range(1, len(epochs)):
        if int(epochs[i - 1]) != int(epochs[i]):
            ax1.axvline(x=i + 1, color="gray", linestyle="--", alpha=0.25)
            drew_epoch_lines = True

    ax1.set_title("Loss (holistic + recent) • Best, trend, ideal zone")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")

    # On-chart summary (no LR, no grade)
    summary_txt = (
        f"avg {y_mean:.4f} | med {y_med:.4f} | std {y_std:.4f}\n"
        f"best {best_loss:.4f} @ {best_idx+1} | avgΔ vs start {avg_improve_pct:.1f}% | variance {session_drop_pct:.1f}%\n"
        f"global slope {global_slope_norm:+.2f}/100"
    )
    ax1.text(
        0.99, 0.99, summary_txt,
        transform=ax1.transAxes,
        va="top", ha="right", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6, edgecolor="none")
    )

    # -------- Legend (outside, bottom-left footer) --------
    loss_color = loss_line.get_color()
    smooth_color = smooth_line.get_color()

    handles = [
        Line2D([0], [0], color=loss_color,  alpha=0.25, label="Loss (raw)"),
        Line2D([0], [0], color=smooth_color,            label=f"Loss (smoothed, k={int(smoothing_k)})"),
        Line2D([0], [0], marker="o", color="green", linestyle="None", markersize=6, label="Best (global min)"),
        Patch(facecolor="green", alpha=0.08, label="Ideal zone (best…best+0.05)"),
    ]
    if outlier_present:
        handles.append(Patch(facecolor="red", alpha=0.05, label="Outlier highlight (|residual| > 1.5σ)"))
    if drew_epoch_lines:
        handles.append(Line2D([0], [0], color="gray", linestyle="--", alpha=0.5, label="Epoch boundary"))

    ncols = 2 if len(handles) > 6 else 1
    footer = 0.18 if ncols == 2 else 0.14
    fig.tight_layout(rect=[0, footer, 1, 1])
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
        framealpha=0.95, fancybox=True, borderpad=0.4,
        handlelength=2.2, handletextpad=0.6,
        ncol=ncols,
    )

    plt.show()


# ----------------------------
# GUI helpers
# ----------------------------
def load_jsonl_file():
    """Prompt for a .jsonl file and load its contents into the input box."""
    global input_box
    file_path = filedialog.askopenfilename(filetypes=[("JSONL files", "*.jsonl")])
    if not file_path or input_box is None:
        return
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()
        input_box.delete("1.0", "end")
        input_box.insert("1.0", raw)
    except Exception as e:
        messagebox.showerror("Error", str(e))

def copy_metrics_to_clipboard():
    """Copy current metrics table into clipboard as 'Metric | Value' lines."""
    global output_frame
    if output_frame is None:
        return
    rows = []
    # children are rendered as alternating labels (metric, value)
    children = [w for w in output_frame.winfo_children() if isinstance(w, ttk.Label)]
    for idx in range(0, len(children), 2):
        metric_key = children[idx].cget("text")
        metric_val = children[idx + 1].cget("text") if idx + 1 < len(children) else ""
        rows.append(f"{metric_key}: {metric_val}")
    text = "\n".join(rows)
    if _HAS_PYPERCLIP:
        try:
            pyperclip.copy(text)
            messagebox.showinfo("Copied", "Metrics copied to clipboard.")
        except Exception as e:
            messagebox.showwarning("Clipboard", f"Could not copy: {e}")
    else:
        # Fallback: put it back into the input box so user can copy
        if input_box is not None:
            input_box.insert("end", "\n\n# Metrics\n" + text + "\n")
            input_box.see("end")
        messagebox.showinfo("Clipboard", "pyperclip not installed; metrics appended to the input box.")

def export_to_markdown():
    """Export the metrics table as a Markdown file."""
    global output_frame
    if output_frame is None:
        return
    children = [w for w in output_frame.winfo_children() if isinstance(w, ttk.Label)]
    lines = ["| Metric | Value |", "|--------|--------|"]
    for idx in range(0, len(children), 2):
        metric_key = children[idx].cget("text")
        metric_val = children[idx + 1].cget("text") if idx + 1 < len(children) else ""
        lines.append(f"| {metric_key} | {metric_val} |")
    save_path = filedialog.asksaveasfilename(defaultextension=".md", filetypes=[("Markdown", "*.md")])
    if save_path:
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            messagebox.showinfo("Saved", f"Saved to {save_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

# ----------------------------
# GUI setup
# ----------------------------
def launch_gui():
    global input_box, output_frame

    root = tk.Tk()
    root.title("Loss Curve Analyzer")
    root.geometry("900x640")

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    mainframe = ttk.Frame(root, padding=10)
    mainframe.pack(fill="both", expand=True)

    # Input box
    input_box = ScrolledText(mainframe, width=120, height=16, font=("Consolas", 11), wrap="none")
    input_box.grid(row=0, column=0, pady=(6, 10), sticky="nsew")

    # Output metrics
    output_frame = ttk.Frame(mainframe)
    output_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))

    # Buttons
    button_frame = ttk.Frame(mainframe)
    button_frame.grid(row=2, column=0, sticky="ew")

    # --- Handlers ---
    def handle_analyze(event=None):
        raw = input_box.get("1.0", "end").strip()
        logs = parse_loss_input(raw)
        if not logs:
            messagebox.showwarning("No logs", "Paste some JSON/JSONL training rows first.")
            return
        plot_loss_curve(logs)

    ttk.Button(button_frame, text="Load JSONL", command=load_jsonl_file).pack(side="left", padx=4)
    ttk.Button(button_frame, text="Copy Metrics", command=copy_metrics_to_clipboard).pack(side="left", padx=4)
    ttk.Button(button_frame, text="Export Markdown", command=export_to_markdown).pack(side="left", padx=4)
    ttk.Button(button_frame, text="Analyze Now", command=handle_analyze).pack(side="left", padx=4)

    # Keyboard shortcut: Ctrl+Enter to analyze
    root.bind("<Control-Return>", handle_analyze)

    # Resizing behavior
    mainframe.rowconfigure(0, weight=1)
    mainframe.rowconfigure(1, weight=0)
    mainframe.columnconfigure(0, weight=1)

    root.mainloop()

# ----------------------------
# Entrypoint
# ----------------------------
def _cli_main():
    if len(sys.argv) > 1 and sys.argv[1].lower().endswith(".jsonl"):
        path = sys.argv[1]
        if not os.path.exists(path):
            print(f"File not found: {path}")
            sys.exit(2)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        logs = parse_loss_input(raw)
        if logs:
            plot_loss_curve(logs)
        else:
            print("No valid log rows found.")
    else:
        launch_gui()

if __name__ == "__main__":
    _cli_main()
