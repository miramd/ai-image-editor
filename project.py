import os
import cv2
import numpy as np
from tkinter import (
    Tk,
    Label,
    Scale,
    HORIZONTAL,
    Button,
    Frame,
    LEFT,
    RIGHT,
    TOP,
    X,
    Y,
    BOTH,
    Canvas,
    NW,
    Entry,
    StringVar,
    Menu,
)
from tkinter import messagebox
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import ImageEnhance, ImageFilter

try:
    from PIL import Image, ImageTk, ImageGrab
except Exception as exc:
    raise SystemExit(
        "Pillow is required for the Tk UI. Install with: pip install pillow"
    ) from exc

try:
    from rembg import remove
except Exception as exc:
    raise SystemExit(
        "rembg is required for background removal. Install with: pip install rembg"
    ) from exc

try:
    from simple_lama_inpainting import SimpleLama
except Exception:
    SimpleLama = None

root = Tk()
root.title("Enhanced Image")
root.configure(bg="#0b111b")
root.update_idletasks()
sw = root.winfo_screenwidth()
sh = root.winfo_screenheight()
win_w = max(980, min(1380, int(sw * 0.9)))
win_h = max(620, min(860, int(sh * 0.88)))
root.geometry(f"{win_w}x{win_h}")
root.minsize(920, 580)
root.attributes("-topmost", True)
root.update_idletasks()
root.attributes("-topmost", False)
root.lift()
root.focus_force()

# State
original_bgr = None
source_bgr = None
processed_pil = None
remove_bg = False
custom_bg_pil = None
ai_erase_active = False
drag_start = None
drag_rect_id = None
after_preview_size = None
simple_lama = None
undo_stack = []
redo_stack = []
resize_w_var = StringVar()
resize_h_var = StringVar()
status_text = StringVar()
resize_target = None
resize_ui_lock = False
after_preview_origin = (0, 0)
after_photo_size = None
after_preview_scale = (1.0, 1.0)
zoom_percent = 100
pan_offset = (0, 0)
pan_start = None
zoom_tip_shown = False
resize_drag_start = None
resize_zoom_start = None
resize_tip_shown = False
zoom_select_active = False
zoom_select_start = None
zoom_select_rect_id = None
move_mode = None
move_drag_start = None
move_undo_snapshot = None
fg_offset = (0, 0)
bg_offset = (0, 0)
fg_scale = 1.0
bg_scale = 1.0
blend_scale = None
blend_active = False
cached_fg_cutout = None
cached_fg_params = None
crop_active = False
crop_start = None
crop_rect_id = None
crop_outline_id = None
slider_job = None
resize_job = None
slider_dragging = False
last_state = None
suppress_undo = False
denoise_mode = "natural"
denoise_controls_enabled = False
denoise_mode_menu = None
preview_layout_job = None
denoise_base_pil = None

# Preview sizing
MAX_PREVIEW_W = 520
MAX_PREVIEW_H = 300

# Theme definitions
# we maintain separate dictionaries for dark and light modes and swap between them
DARK_THEME = {
    "bg": "#0b111b",
    "panel": "#121a27",
    "card": "#0f1724",
    "border": "#243447",
    "accent": "#3b82f6",
    "accent_hover": "#0b4ed0",
    "accent_2": "#f59e0b",
    "text": "#e5edf7",
    "muted": "#8aa0bb",
    "canvas_bg": "#0a1220",
    "entry_bg": "#0d1625",
    "button_active": "#0f766e",
    "save_button": "#059669",
    "save_button_active": "#047857",
    "bg_choice_button": "#0ea5e9",
    "bg_choice_hover": "#0284c7",
}

LIGHT_THEME = {
    "bg": "#ffffff",
    "panel": "#f3f4f6",
    "card": "#ffffff",
    "border": "#d1d5db",
    "accent": "#3b82f6",
    "accent_hover": "#bfdbfe",
    "accent_2": "#f59e0b",
    "text": "#111827",
    "muted": "#6b7280",
    "canvas_bg": "#ffffff",
    "entry_bg": "#ffffff",
    "button_active": "#bfdbfe",
    "save_button": "#10b981",
    "save_button_active": "#059669",
    "bg_choice_button": "#0ea5e9",
    "bg_choice_hover": "#0284c7",
}

# start in dark mode
THEME = DARK_THEME
current_theme = "dark"

# toolbar colours are derived from THEME, update after THEME assignment
TOOLBAR_BG = THEME["panel"]
TOOLBAR_HOVER = THEME.get("accent_hover", "#1f2d3e")

# helper to convert old theme values to new ones for existing widgets

def _replace_color(val, old_theme, new_theme):
    for k, v in old_theme.items():
        if val == v:
            return new_theme.get(k, val)
    return val


def apply_theme():
    """Walk the widget tree and replace any colours that match the old theme.

    This relies on most widgets having been initialised with values taken from
    the THEME dictionary. When the user toggles modes we swap THEME and then
    call this function to update every widget in place.
    """
    old = LIGHT_THEME if current_theme == "dark" else DARK_THEME
    new = THEME

    def recurse(w):
        # only configure options that the widget actually exposes
        for opt in ("bg", "fg", "highlightbackground", "activebackground", "activeforeground", "insertbackground", "troughcolor"):
            try:
                if opt in w.keys():
                    cur = w.cget(opt)
                    if cur:
                        updated = _replace_color(cur, old, new)
                        if updated != cur:
                            w.configure(**{opt: updated})
            except Exception:
                pass
        for c in w.winfo_children():
            recurse(c)

    recurse(root)
    # some canvases are reconfigured elsewhere; make sure they pick up the new value
    try:
        before_canvas.configure(bg=THEME["canvas_bg"])
        after_canvas.configure(bg=THEME["canvas_bg"])
    except NameError:
        pass

    # reapply hover styling for all buttons so hover colours update with the theme
    try:
        apply_hover(upload_button, THEME["accent"], THEME.get("accent_hover"))
        apply_hover(save_button, THEME.get("save_button"), THEME.get("save_button_active"))
        apply_hover(screen_button, THEME.get("accent", "#6366f1"), THEME.get("accent_hover", "#4f46e5"))
        apply_hover(remove_bg_button, THEME.get("accent_2"), "#d97706")
        apply_hover(bg_choice_button, THEME.get("bg_choice_button"), THEME.get("bg_choice_hover"))
        apply_hover(blend_button, THEME.get("bg_choice_button"), THEME.get("bg_choice_hover"))
        apply_hover(reset_button, THEME.get("border", "#475569"), THEME.get("panel", "#334155"))
        apply_hover(ai_erase_button, "#f59e0b", "#d97706")
        apply_hover(denoise_mode_button, THEME.get("border"), THEME.get("panel"))
        apply_hover(zoom_select_button, THEME.get("border"), THEME.get("panel"))
        apply_hover(undo_button, THEME.get("border"), THEME.get("panel"))
        apply_hover(redo_button, THEME.get("border"), THEME.get("panel"))
        apply_hover(fit_screen_button, THEME.get("border"), THEME.get("panel"))
        apply_hover(zoom_out_button, THEME.get("border"), THEME.get("panel"))
        apply_hover(zoom_in_button, THEME.get("border"), THEME.get("panel"))
        apply_hover(rotate_button, THEME.get("border"), THEME.get("panel"))
        apply_hover(crop_button, THEME.get("border"), THEME.get("panel"))
        apply_hover(move_fg_button, THEME.get("accent"), THEME.get("accent_hover"))
        apply_hover(move_bg_button, THEME.get("accent"), THEME.get("accent_hover"))
        apply_hover(denoise_button, THEME.get("accent_2"), THEME.get("accent_2"))
    except NameError:
        # some buttons may not exist yet when apply_theme is invoked early
        pass


def toggle_theme():
    global current_theme, THEME, TOOLBAR_BG, TOOLBAR_HOVER
    if current_theme == "dark":
        current_theme = "light"
        THEME = LIGHT_THEME
    else:
        current_theme = "dark"
        THEME = DARK_THEME
    TOOLBAR_BG = THEME["panel"]
    TOOLBAR_HOVER = THEME.get("accent_hover", TOOLBAR_HOVER)
    # update button label as well
    theme_button.config(text="Dark Mode" if current_theme == "light" else "Light Mode")
    apply_theme()

# initial widgets are created using THEME values, so no need to call apply_theme here
FONT_TITLE = ("Segoe UI Semibold", 14)
FONT_LABEL = ("Segoe UI Semibold", 10)
FONT_BODY = ("Segoe UI", 10)

# Header
header_bar = Frame(root, bg=THEME["bg"])
header_bar.pack(fill=X, padx=14, pady=(12, 6))
Label(
    header_bar,
    text="Image Studio",
    font=("Segoe UI Semibold", 18),
    fg=THEME["text"],
    bg=THEME["bg"],
).pack(anchor="w")
Label(
    header_bar,
    text="Modern editor: clear previews, grouped tools, fast workflow",
    font=("Segoe UI", 10),
    fg=THEME["muted"],
    bg=THEME["bg"],
).pack(anchor="w", pady=(2, 0))

# theme toggle switch (top right corner)
theme_button = Button(
    header_bar,
    text="Light Mode",
    command=toggle_theme,
    bg=THEME["panel"],
    fg=THEME["text"],
    bd=0,
    highlightthickness=0,
    relief="flat",
    cursor="hand2",
)
theme_button.pack(side=RIGHT, padx=4, pady=3)

# Main workspace
workspace = Frame(root, bg=THEME["bg"])
workspace.pack(fill=BOTH, expand=True, padx=14, pady=(0, 12))

# Toolbar container (horizontal)
toolbar_shell = Frame(
    workspace,
    bg=THEME["panel"],
    bd=0,
    highlightthickness=1,
    highlightbackground=THEME["border"],
)
toolbar_shell.pack(side=TOP, fill=X, padx=0, pady=0)
toolbar_top = Frame(toolbar_shell, bg=TOOLBAR_BG)
toolbar_top.pack(fill=X, padx=8, pady=(8, 6))
toolbar_bottom = Frame(toolbar_shell, bg=TOOLBAR_BG)
toolbar_bottom.pack(fill=X, padx=8, pady=(0, 8))

toolbar_top_inner = Frame(toolbar_top, bg=TOOLBAR_BG)
toolbar_top_inner.pack(fill=X)
toolbar_bottom_inner = Frame(toolbar_bottom, bg=TOOLBAR_BG)
toolbar_bottom_inner.pack(fill=X)

group_card = {
    "bg": TOOLBAR_BG,
    "bd": 0,
    "highlightthickness": 1,
    "highlightbackground": THEME["border"],
}
file_group = Frame(toolbar_top_inner, **group_card)
edit_group = Frame(toolbar_top_inner, **group_card)
tools_group = Frame(toolbar_bottom_inner, **group_card)

file_group.pack(side=LEFT, padx=(0, 10), pady=(0, 0))
edit_group.pack(side=LEFT, padx=(0, 10), pady=(0, 0))
tools_group.pack(side=LEFT, padx=(0, 0), pady=(0, 0))

Label(file_group, text="FILE", font=FONT_LABEL, fg=THEME["muted"], bg=TOOLBAR_BG).pack(anchor="w", padx=10, pady=(6, 0))
Label(edit_group, text="EDIT", font=FONT_LABEL, fg=THEME["muted"], bg=TOOLBAR_BG).pack(anchor="w", padx=10, pady=(6, 0))
Label(tools_group, text="TOOLS", font=FONT_LABEL, fg=THEME["muted"], bg=TOOLBAR_BG).pack(anchor="w", padx=10, pady=(6, 0))

file_row = Frame(file_group, bg=TOOLBAR_BG)
edit_row = Frame(edit_group, bg=TOOLBAR_BG)
tools_row = Frame(tools_group, bg=TOOLBAR_BG)
file_row.pack(fill=X, padx=8, pady=(4, 8))
edit_row.pack(fill=X, padx=8, pady=(4, 8))
tools_row.pack(fill=X, padx=8, pady=(4, 8))

# Layout: app container
app = Frame(workspace, bg=THEME["bg"])
app.pack(side=TOP, fill=BOTH, expand=True)

# Scrollable left panel
left_panel = Frame(app, bg=THEME["panel"], width=320, bd=0, highlightthickness=1, highlightbackground=THEME["border"])
left_panel.pack(side=LEFT, fill=Y, padx=12, pady=12)
left_panel.pack_propagate(False)

left_canvas = Canvas(left_panel, bg=THEME["panel"], highlightthickness=0)
left_scroll = Frame(left_panel, bg=THEME["panel"])
left_scroll.pack(side=RIGHT, fill=Y)

scrollbar = None
try:
    from tkinter import Scrollbar
    scrollbar = Scrollbar(left_scroll, orient="vertical", command=left_canvas.yview)
    scrollbar.pack(fill=Y)
    left_canvas.configure(yscrollcommand=scrollbar.set)
except Exception:
    pass

left_canvas.pack(side=LEFT, fill=BOTH, expand=True)

left_inner = Frame(left_canvas, bg=THEME["panel"])
left_window = left_canvas.create_window((0, 0), window=left_inner, anchor="nw")

def update_left_scroll(_=None):
    left_canvas.configure(scrollregion=left_canvas.bbox("all"))
    left_canvas.itemconfig(left_window, width=left_canvas.winfo_width())

left_inner.bind("<Configure>", update_left_scroll)
left_canvas.bind("<Configure>", update_left_scroll)

right_panel = Frame(app, bg=THEME["bg"])
right_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=12, pady=12)

# UI: previews
preview_row = Frame(right_panel, bg=THEME["bg"])
preview_row.pack(padx=0, pady=0, fill=BOTH, expand=True)
preview_row.pack_propagate(False)

before_frame = Frame(
    preview_row,
    bg=THEME["card"],
    bd=0,
    highlightthickness=1,
    highlightbackground=THEME["border"],
)
before_frame.pack(side=LEFT, padx=(0, 8), pady=0, anchor="n", expand=True, fill=BOTH)
Label(before_frame, text="Before", font=FONT_LABEL, fg=THEME["muted"], bg=THEME["card"]).pack(anchor="w", padx=12, pady=(10, 6))
before_canvas = Canvas(
    before_frame,
    highlightthickness=0,
    bg=THEME["canvas_bg"],
    width=MAX_PREVIEW_W,
    height=MAX_PREVIEW_H,
)
before_canvas.pack(fill=BOTH, expand=True, padx=12, pady=(0, 12))

after_frame = Frame(
    preview_row,
    bg=THEME["card"],
    bd=0,
    highlightthickness=1,
    highlightbackground=THEME["border"],
)
after_frame.pack(side=LEFT, padx=(8, 0), pady=0, anchor="n", expand=True, fill=BOTH)
Label(after_frame, text="After", font=FONT_LABEL, fg=THEME["muted"], bg=THEME["card"]).pack(anchor="w", padx=12, pady=(10, 6))
after_canvas = Canvas(
    after_frame,
    highlightthickness=0,
    bg=THEME["canvas_bg"],
    width=MAX_PREVIEW_W,
    height=MAX_PREVIEW_H,
)
after_canvas.pack(fill=BOTH, expand=True, padx=12, pady=(0, 12))


def refresh_preview_layout(_event=None):
    global preview_layout_job
    if preview_layout_job is not None:
        root.after_cancel(preview_layout_job)
    preview_layout_job = root.after(80, apply_preview_layout)


def apply_preview_layout():
    global MAX_PREVIEW_W, MAX_PREVIEW_H, preview_layout_job
    preview_layout_job = None
    panel_w = max(380, right_panel.winfo_width() - 16)
    panel_h = max(320, right_panel.winfo_height() - 16)
    target_w = max(260, min(760, (panel_w - 64) // 2))
    target_h = max(240, min(760, panel_h - 90))
    if target_w == MAX_PREVIEW_W and target_h == MAX_PREVIEW_H:
        return
    MAX_PREVIEW_W = target_w
    MAX_PREVIEW_H = target_h
    before_canvas.configure(width=MAX_PREVIEW_W, height=MAX_PREVIEW_H)
    after_canvas.configure(width=MAX_PREVIEW_W, height=MAX_PREVIEW_H)
    if original_bgr is not None:
        render_previews()

# UI: controls
controls = Frame(
    left_inner,
    bg=THEME["panel"],
    bd=0,
    highlightthickness=1,
    highlightbackground=THEME["border"],
)
controls.pack(fill=X, padx=10, pady=10)

Label(controls, text="Editing Panel", font=FONT_TITLE, fg=THEME["text"], bg=THEME["panel"]).pack(anchor="w", padx=10, pady=(10, 2))
Label(controls, text="Image size and tone", font=("Segoe UI", 9), fg=THEME["muted"], bg=THEME["panel"]).pack(anchor="w", padx=10, pady=(0, 6))

Label(
    controls,
    text="Resize",
    font=FONT_LABEL,
    fg=THEME["accent"],
    bg=THEME["panel"],
).pack(anchor="w", pady=(6, 4))

resize_row = Frame(controls, bg=THEME["panel"])
resize_row.pack(fill=X, padx=10)

Label(resize_row, text="W", font=FONT_BODY, fg=THEME["text"], bg=THEME["panel"]).pack(side=LEFT, padx=(0, 4))
resize_w_entry = Entry(
    resize_row,
    textvariable=resize_w_var,
    width=6,
    bg=THEME.get("entry_bg"),
    fg=THEME["text"],
    insertbackground=THEME["text"],
    relief="flat",
)
resize_w_entry.pack(side=LEFT, padx=(0, 10))

Label(resize_row, text="H", font=FONT_BODY, fg=THEME["text"], bg=THEME["panel"]).pack(side=LEFT, padx=(0, 4))
resize_h_entry = Entry(
    resize_row,
    textvariable=resize_h_var,
    width=6,
    bg=THEME.get("entry_bg"),
    fg=THEME["text"],
    insertbackground=THEME["text"],
    relief="flat",
)
resize_h_entry.pack(side=LEFT, padx=(0, 10))

apply_wh_button = Button(
    resize_row,
    text="Apply",
    bg=THEME["accent"],
    fg="white",
    activebackground=THEME.get("button_active"),
    activeforeground="white",
    bd=0,
    highlightthickness=0,
    relief="flat",
    font=FONT_BODY,
    padx=8,
    pady=2,
    cursor="hand2",
)
apply_wh_button.pack(side=RIGHT)

percent_scale = Scale(
    controls,
    from_=10,
    to=400,
    orient=HORIZONTAL,
    label="Resize (%)",
    bg=THEME["panel"],
    fg=THEME["text"],
    troughcolor=THEME.get("border"),
    highlightthickness=0,
    font=FONT_BODY,
)
percent_scale.set(100)
percent_scale.pack(fill=X, padx=10)

apply_percent_button = Button(
    controls,
    text="Apply Resize",
    bg=THEME["accent"],
    fg="white",
    activebackground=THEME.get("button_active"),
    activeforeground="white",
    bd=0,
    highlightthickness=0,
    relief="flat",
    font=FONT_BODY,
    padx=8,
    pady=4,
    cursor="hand2",
)
apply_percent_button.pack(fill=X, padx=10, pady=(4, 6))

Label(controls, text="Color and detail", font=("Segoe UI Semibold", 9), fg=THEME["muted"], bg=THEME["panel"]).pack(anchor="w", padx=10, pady=(2, 2))

contrast_scale = Scale(
    controls,
    from_=0,
    to=30,
    orient=HORIZONTAL,
    label="Contrast",
    bg=THEME["panel"],
    fg=THEME["text"],
    troughcolor=THEME.get("border"),
    highlightthickness=0,
    font=FONT_BODY,
)
contrast_scale.set(10)
contrast_scale.pack(fill=X, padx=10)

brightness_scale = Scale(
    controls,
    from_=0,
    to=100,
    orient=HORIZONTAL,
    label="Brightness",
    bg=THEME["panel"],
    fg=THEME["text"],
    troughcolor=THEME.get("border"),
    highlightthickness=0,
    font=FONT_BODY,
)
brightness_scale.set(0)
brightness_scale.pack(fill=X, padx=10)

saturation_scale = Scale(
    controls,
    from_=0,
    to=200,
    orient=HORIZONTAL,
    label="Color (Saturation %)",
    bg=THEME["panel"],
    fg=THEME["text"],
    troughcolor=THEME.get("border"),
    highlightthickness=0,
    font=FONT_BODY,
)
saturation_scale.set(100)
saturation_scale.pack(fill=X, padx=10)

clarity_scale = Scale(
    controls,
    from_=-20,
    to=20,
    orient=HORIZONTAL,
    label="Clarity (Blur / Sharpen)",
    bg=THEME["panel"],
    fg=THEME["text"],
    troughcolor=THEME.get("border"),
    highlightthickness=0,
    font=FONT_BODY,
)
clarity_scale.set(0)
clarity_scale.pack(fill=X, padx=10)

denoise_scale = Scale(
    controls,
    from_=0,
    to=80,
    orient=HORIZONTAL,
    label="Denoise Strength",
    bg=THEME["panel"],
    fg=THEME["text"],
    troughcolor=THEME.get("border"),
    highlightthickness=0,
    font=FONT_BODY,
)
denoise_scale.set(28)
denoise_scale.pack(fill=X, padx=10)
denoise_scale.configure(state="disabled")

blend_scale = Scale(
    controls,
    from_=0,
    to=100,
    orient=HORIZONTAL,
    label="Blend Strength (%)",
    bg=THEME["panel"],
    fg=THEME["text"],
    troughcolor=THEME.get("border"),
    highlightthickness=0,
    font=FONT_BODY,
)
blend_scale.set(100)
blend_scale.pack(fill=X, padx=10, pady=(0, 10))

def apply_hover(button, normal_bg, hover_bg):
    button.configure(bg=normal_bg, activebackground=hover_bg)
    button.bind("<Enter>", lambda _e: button.configure(bg=hover_bg))
    button.bind("<Leave>", lambda _e: button.configure(bg=normal_bg))


button_style = {
    "bd": 0,
    "highlightthickness": 0,
    "relief": "flat",
    "font": FONT_BODY,
    "padx": 12,
    "pady": 6,
    "cursor": "hand2",
    "width": 10,
}
tools_button_style = dict(button_style)
tools_button_style["width"] = 9

upload_button = Button(
    file_row,
    text="Upload",
    fg="white",
    activeforeground="white",
    **button_style,
)
upload_button.pack(side=LEFT, padx=4, pady=3)

remove_bg_button = Button(
    edit_row,
    text="BG: OFF",
    fg="white",
    activeforeground="white",
    **button_style,
)
remove_bg_button.pack(side=LEFT, padx=4, pady=3)

bg_choice_button = Button(
    edit_row,
    text="Background",
    fg="white",
    activeforeground="white",
    **button_style,
)
bg_choice_button.pack(side=LEFT, padx=4, pady=3)

blend_button = Button(
    edit_row,
    text="Blend",
    fg="white",
    activeforeground="white",
    **button_style,
)
blend_button.pack(side=LEFT, padx=4, pady=3)

reset_button = Button(
    edit_row,
    text="Reset",
    fg="#f1f3f5",
    activeforeground="#f1f3f5",
    **button_style,
)
reset_button.pack(side=LEFT, padx=4, pady=3)

screen_button = Button(
    file_row,
    text="Capture",
    fg="#f1f3f5",
    activeforeground="#f1f3f5",
    **button_style,
)
screen_button.pack(side=LEFT, padx=4, pady=3)

save_button = Button(
    file_row,
    text="Save",
    fg="white",
    activeforeground="white",
    **button_style,
)
save_button.pack(side=LEFT, padx=4, pady=3)

ai_erase_button = Button(
    tools_row,
    text="AI Erase",
    fg="#1a1a1a",
    activeforeground="#1a1a1a",
    **tools_button_style,
)
ai_erase_button.pack(side=LEFT, padx=4, pady=3)

zoom_select_button = Button(
    tools_row,
    text="Zoom Sel",
    fg="white",
    activeforeground="white",
    **tools_button_style,
)
zoom_select_button.pack(side=LEFT, padx=4, pady=3)

undo_button = Button(
    tools_row,
    text="Undo",
    fg="white",
    activeforeground="white",
    **tools_button_style,
)
undo_button.pack(side=LEFT, padx=4, pady=3)

redo_button = Button(
    tools_row,
    text="Redo",
    fg="white",
    activeforeground="white",
    **tools_button_style,
)
redo_button.pack(side=LEFT, padx=4, pady=3)

fit_screen_button = Button(
    tools_row,
    text="Fit",
    fg="white",
    activeforeground="white",
    **tools_button_style,
)
fit_screen_button.pack(side=LEFT, padx=4, pady=3)

zoom_out_button = Button(
    tools_row,
    text="Zoom -",
    fg="white",
    activeforeground="white",
    **tools_button_style,
)
zoom_out_button.pack(side=LEFT, padx=4, pady=3)

zoom_in_button = Button(
    tools_row,
    text="Zoom +",
    fg="white",
    activeforeground="white",
    **tools_button_style,
)
zoom_in_button.pack(side=LEFT, padx=4, pady=3)

rotate_button = Button(
    tools_row,
    text="Rotate",
    fg="white",
    activeforeground="white",
    **tools_button_style,
)
rotate_button.pack(side=LEFT, padx=4, pady=3)

crop_button = Button(
    tools_row,
    text="Crop",
    fg="white",
    activeforeground="white",
    **tools_button_style,
)
crop_button.pack(side=LEFT, padx=4, pady=3)

move_fg_button = Button(
    tools_row,
    text="Move FG",
    fg="white",
    activeforeground="white",
    **tools_button_style,
)
move_fg_button.pack(side=LEFT, padx=4, pady=3)

move_bg_button = Button(
    tools_row,
    text="Move BG",
    fg="white",
    activeforeground="white",
    **tools_button_style,
)
move_bg_button.pack(side=LEFT, padx=4, pady=3)

denoise_button = Button(
    tools_row,
    text="Denoise",
    fg="white",
    activeforeground="white",
    **tools_button_style,
)
denoise_button.pack(side=LEFT, padx=4, pady=3)

denoise_mode_button = Button(
    tools_row,
    text="Mode: Nat",
    fg="white",
    activeforeground="white",
    **dict(tools_button_style, width=10),
)
denoise_mode_button.pack(side=LEFT, padx=4, pady=3)

apply_hover(upload_button, THEME["accent"], THEME.get("accent_hover"))
apply_hover(save_button, THEME.get("save_button"), THEME.get("save_button_active"))
apply_hover(screen_button, THEME.get("accent", "#6366f1"), THEME.get("accent_hover", "#4f46e5"))
apply_hover(remove_bg_button, THEME["accent_2"], "#d97706")
apply_hover(bg_choice_button, THEME.get("bg_choice_button"), THEME.get("bg_choice_hover"))
apply_hover(blend_button, THEME.get("bg_choice_button"), THEME.get("bg_choice_hover"))
apply_hover(reset_button, THEME.get("border", "#475569"), THEME.get("panel", "#334155"))
apply_hover(ai_erase_button, THEME.get("accent_2", "#f59e0b"), "#d97706")
apply_hover(denoise_mode_button, THEME.get("border", "#243447"), THEME.get("panel", "#1c2a3a"))
apply_hover(zoom_select_button, THEME.get("border", "#243447"), THEME.get("panel", "#1c2a3a"))
apply_hover(undo_button, THEME.get("border", "#475569"), THEME.get("panel", "#334155"))
apply_hover(redo_button, THEME.get("border", "#475569"), THEME.get("panel", "#334155"))
apply_hover(fit_screen_button, THEME.get("border", "#243447"), THEME.get("panel", "#1c2a3a"))
apply_hover(zoom_out_button, THEME.get("border", "#243447"), THEME.get("panel", "#1c2a3a"))
apply_hover(zoom_in_button, THEME.get("border", "#243447"), THEME.get("panel", "#1c2a3a"))
apply_hover(rotate_button, THEME.get("border", "#243447"), THEME.get("panel", "#1c2a3a"))
apply_hover(crop_button, THEME.get("border", "#243447"), THEME.get("panel", "#1c2a3a"))
apply_hover(move_fg_button, THEME.get("accent", "#2563eb"), THEME.get("accent_hover", "#1d4ed8"))
apply_hover(move_bg_button, THEME.get("accent", "#2563eb"), THEME.get("accent_hover", "#1d4ed8"))
apply_hover(denoise_button, THEME.get("accent_2", "#7c3aed"), THEME.get("accent_2", "#6d28d9"))


def set_controls_enabled(enabled):
    state = "normal" if enabled else "disabled"
    for widget in (
        remove_bg_button,
        bg_choice_button,
        blend_button,
        reset_button,
        screen_button,
        save_button,
        ai_erase_button,
        zoom_select_button,
        undo_button,
        redo_button,
        fit_screen_button,
        zoom_out_button,
        zoom_in_button,
        rotate_button,
        crop_button,
        move_fg_button,
        move_bg_button,
        denoise_button,
        apply_wh_button,
        apply_percent_button,
        percent_scale,
        contrast_scale,
        brightness_scale,
        saturation_scale,
        clarity_scale,
        blend_scale,
        resize_w_entry,
        resize_h_entry,
    ):
        widget.configure(state=state)
    update_blend_controls()


def update_blend_controls():
    state = "normal" if (blend_active and remove_bg and custom_bg_pil is not None) else "disabled"
    move_fg_button.configure(state=state)
    move_bg_button.configure(state=state)

# Disable editing controls until an image is uploaded
set_controls_enabled(False)


def fit_preview(pil_img, zoom_pct, cap_size=True):
    src_w, src_h = pil_img.size
    if src_w <= 0 or src_h <= 0:
        return pil_img.copy()

    # Compute scale from original each time to avoid zooming an already-downscaled preview.
    if cap_size:
        fit_scale = min(MAX_PREVIEW_W / src_w, MAX_PREVIEW_H / src_h)
        fit_scale = min(fit_scale, 1.0)
    else:
        fit_scale = 1.0
    scale = fit_scale * (zoom_pct / 100.0)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    return pil_img.resize((new_w, new_h), Image.LANCZOS)


def render_previews():
    if original_bgr is None:
        return

    # Before (always show the original uploaded image when available)
    base_bgr = source_bgr if source_bgr is not None else original_bgr
    before_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
    before_pil = Image.fromarray(before_rgb)
    before_preview = fit_preview(before_pil, 100, cap_size=True)
    # After (zoomed preview only)
    after_preview = fit_preview(processed_pil, zoom_percent, cap_size=True)

    before_tk = ImageTk.PhotoImage(before_preview)
    before_canvas.configure(bg=THEME["canvas_bg"])    
    before_canvas.delete("all")
    bx = (MAX_PREVIEW_W - before_preview.width) // 2
    by = (MAX_PREVIEW_H - before_preview.height) // 2
    before_canvas.create_image(bx, by, anchor=NW, image=before_tk)
    before_canvas.image = before_tk

    # After
    after_tk = ImageTk.PhotoImage(after_preview)
    after_canvas.configure(bg=THEME["canvas_bg"])    
    after_canvas.delete("all")
    ax = (MAX_PREVIEW_W - after_preview.width) // 2
    ay = (MAX_PREVIEW_H - after_preview.height) // 2
    after_canvas.create_image(ax, ay, anchor=NW, image=after_tk, tags=("after_img",))
    after_canvas.image = after_tk
    global after_preview_size, after_preview_origin, pan_offset, after_photo_size, after_preview_scale
    after_preview_size = (after_preview.width, after_preview.height)
    after_photo_size = (after_tk.width(), after_tk.height())
    if processed_pil is not None and processed_pil.width and processed_pil.height:
        after_preview_scale = (
            after_preview.width / processed_pil.width,
            after_preview.height / processed_pil.height,
        )
    else:
        after_preview_scale = (1.0, 1.0)
    pan_offset = (0, 0)
    after_preview_origin = (ax, ay)
    update_status()


def capture_state():
    return {
        "kind": "full",
        "source_bgr": source_bgr.copy() if source_bgr is not None else None,
        "original_bgr": original_bgr.copy() if original_bgr is not None else None,
        "processed_pil": processed_pil.copy() if processed_pil is not None else None,
        "remove_bg": remove_bg,
        "custom_bg_pil": custom_bg_pil.copy() if custom_bg_pil is not None else None,
        "resize_target": resize_target,
        "contrast": contrast_scale.get(),
        "brightness": brightness_scale.get(),
        "saturation": saturation_scale.get(),
        "clarity": clarity_scale.get(),
        "denoise_strength": denoise_scale.get(),
        "denoise_mode": denoise_mode,
        "denoise_controls_enabled": denoise_controls_enabled,
        "blend_strength": blend_scale.get() if blend_scale is not None else 100,
        "fg_offset": fg_offset,
        "bg_offset": bg_offset,
        "fg_scale": fg_scale,
        "bg_scale": bg_scale,
        "blend_active": blend_active,
        "zoom_percent": zoom_percent,
        "pan_offset": pan_offset,
    }


def commit_state():
    global last_state
    last_state = capture_state()


def push_undo_from_last():
    if last_state is None:
        return
    undo_stack.append(last_state)
    if len(undo_stack) > 10:
        undo_stack.pop(0)
    redo_stack.clear()


def restore_state(state):
    global source_bgr, original_bgr, processed_pil, remove_bg, custom_bg_pil
    global resize_target, fg_offset, bg_offset, fg_scale, bg_scale, denoise_mode, denoise_controls_enabled
    global blend_active, zoom_percent, pan_offset, suppress_undo, denoise_base_pil
    if not state:
        return
    suppress_undo = True
    try:
        source_bgr = state.get("source_bgr")
        original_bgr = state.get("original_bgr")
        processed_pil = state.get("processed_pil")
        remove_bg = state.get("remove_bg", False)
        custom_bg_pil = state.get("custom_bg_pil")
        resize_target = state.get("resize_target")
        contrast_scale.set(state.get("contrast", contrast_scale.get()))
        brightness_scale.set(state.get("brightness", brightness_scale.get()))
        saturation_scale.set(state.get("saturation", saturation_scale.get()))
        clarity_scale.set(state.get("clarity", clarity_scale.get()))
        denoise_scale.set(state.get("denoise_strength", denoise_scale.get()))
        denoise_mode = state.get("denoise_mode", denoise_mode)
        denoise_controls_enabled = state.get("denoise_controls_enabled", denoise_controls_enabled)
        denoise_base_pil = None
        update_denoise_mode_button()
        set_denoise_controls_enabled(denoise_controls_enabled)
        if blend_scale is not None:
            blend_scale.set(state.get("blend_strength", blend_scale.get()))
        fg_offset = state.get("fg_offset", (0, 0))
        bg_offset = state.get("bg_offset", (0, 0))
        fg_scale = state.get("fg_scale", 1.0)
        bg_scale = state.get("bg_scale", 1.0)
        blend_active = state.get("blend_active", False)
        zoom_percent = state.get("zoom_percent", zoom_percent)
        pan_offset = state.get("pan_offset", (0, 0))
        remove_bg_button.configure(text="BG: ON" if remove_bg else "BG: OFF")
        if original_bgr is not None:
            h, w = original_bgr.shape[:2]
            update_resize_fields(w, h, 100)
        if processed_pil is None and original_bgr is not None:
            process_image()
        else:
            render_previews()
            if pan_offset != (0, 0):
                after_canvas.move("after_img", pan_offset[0], pan_offset[1])
                global after_preview_origin
                after_preview_origin = (
                    after_preview_origin[0] + pan_offset[0],
                    after_preview_origin[1] + pan_offset[1],
                )
        update_status()
        update_blend_controls()
        commit_state()
    finally:
        suppress_undo = False


def process_image():
    global processed_pil, resize_target, cached_fg_cutout, cached_fg_params, suppress_undo, denoise_base_pil
    if original_bgr is None:
        return
    if not suppress_undo:
        push_undo_from_last()

    contrast = contrast_scale.get() / 10
    brightness = brightness_scale.get()
    saturation = saturation_scale.get() / 100.0
    clarity = clarity_scale.get()

    enhanced = cv2.convertScaleAbs(original_bgr, alpha=contrast, beta=brightness)
    if clarity != 0:
        k = max(1, int(abs(clarity)))
        k = k + 1 if k % 2 == 0 else k
        if clarity < 0:
            enhanced = cv2.GaussianBlur(enhanced, (k, k), 0)
        else:
            blurred = cv2.GaussianBlur(enhanced, (k, k), 0)
            # Unsharp mask
            enhanced = cv2.addWeighted(enhanced, 1.4, blurred, -0.4, 0)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    if saturation != 1.0:
        pil_img = ImageEnhance.Color(pil_img).enhance(saturation)

    if remove_bg:
        cache_params = (
            contrast,
            brightness,
            saturation,
            clarity,
            resize_target,
            original_bgr.shape[:2],
        )
        if blend_active and cached_fg_cutout is not None and cached_fg_params == cache_params:
            pil_img = cached_fg_cutout.copy()
        else:
            pil_img = remove(pil_img).convert("RGBA")
            pil_img = refine_alpha_edges(pil_img)
            if blend_active:
                cached_fg_cutout = pil_img.copy()
                cached_fg_params = cache_params
        if custom_bg_pil is not None:
            fg = pil_img
            bg_base = custom_bg_pil.resize(fg.size, Image.LANCZOS).convert("RGBA")
            bg = transform_layer(bg_base, fg.size, bg_scale, bg_offset)
            fg = transform_layer(fg, fg.size, fg_scale, fg_offset)
            fg = dehalo_foreground(fg, bg)
            # Apply user-controlled blend strength (opacity of foreground).
            strength = max(0, min(100, blend_scale.get() if blend_scale is not None else 100))
            if strength < 100:
                alpha = fg.split()[-1]
                alpha = alpha.point(lambda a: int(a * (strength / 100.0)))
                fg.putalpha(alpha)
            pil_img = Image.alpha_composite(bg, fg)

    if resize_target is not None:
        pil_img = pil_img.resize(resize_target, Image.LANCZOS)
    processed_pil = pil_img
    denoise_base_pil = None
    render_previews()
    if not suppress_undo:
        commit_state()


def on_slider_change(_=None):
    global slider_job
    if slider_job is not None:
        root.after_cancel(slider_job)
    delay = 280 if slider_dragging else 140
    slider_job = root.after(delay, process_image)


def update_resize_fields(width, height, percent=None):
    global resize_ui_lock
    resize_w_var.set(str(width))
    resize_h_var.set(str(height))
    if percent is not None and not resize_ui_lock:
        percent_scale.set(int(percent))


def apply_resize_percent():
    global resize_target, resize_ui_lock
    if original_bgr is None:
        return
    pct = percent_scale.get()
    if pct <= 0:
        return
    h, w = original_bgr.shape[:2]
    new_w = max(1, int(w * pct / 100.0))
    new_h = max(1, int(h * pct / 100.0))
    resize_target = (new_w, new_h)
    process_image()
    resize_ui_lock = True
    try:
        update_resize_fields(new_w, new_h, pct)
    finally:
        resize_ui_lock = False


def apply_resize_wh():
    global resize_target
    if original_bgr is None:
        return
    try:
        new_w = int(resize_w_var.get())
        new_h = int(resize_h_var.get())
    except Exception:
        messagebox.showerror("Resize", "Width and height must be integers.")
        return
    if new_w <= 0 or new_h <= 0:
        messagebox.showerror("Resize", "Width and height must be positive.")
        return
    h, w = original_bgr.shape[:2]
    resize_target = (new_w, new_h)
    process_image()
    pct = round((new_w / w) * 100) if w else 100
    update_resize_fields(new_w, new_h, pct)


def on_percent_change(_=None):
    global resize_job
    if resize_ui_lock:
        return
    if resize_job is not None:
        root.after_cancel(resize_job)
    delay = 320 if slider_dragging else 180
    resize_job = root.after(delay, apply_resize_percent)


def on_slider_press(_event=None):
    global slider_dragging
    slider_dragging = True


def on_slider_release(_event=None):
    global slider_dragging, slider_job
    slider_dragging = False
    if slider_job is not None:
        root.after_cancel(slider_job)
        slider_job = None
    process_image()


def on_percent_release(_event=None):
    global slider_dragging, resize_job
    slider_dragging = False
    if resize_job is not None:
        root.after_cancel(resize_job)
        resize_job = None
    apply_resize_percent()


def on_denoise_change(_=None):
    global slider_job
    if not denoise_controls_enabled or processed_pil is None:
        return
    if slider_job is not None:
        root.after_cancel(slider_job)
    delay = 320 if slider_dragging else 180
    slider_job = root.after(delay, denoise_image)


def on_denoise_release(_event=None):
    global slider_dragging, slider_job
    if not denoise_controls_enabled or processed_pil is None:
        return
    slider_dragging = False
    if slider_job is not None:
        root.after_cancel(slider_job)
        slider_job = None
    denoise_image()


def push_view_undo():
    undo_stack.append(
        {
            "zoom": zoom_percent,
            "pan_offset": pan_offset,
        }
    )
    if len(undo_stack) > 3:
        undo_stack.pop(0)
    redo_stack.clear()


def apply_view_state(state):
    global zoom_percent, pan_offset, after_preview_origin
    zoom_percent = state.get("zoom", zoom_percent)
    pan_offset = state.get("pan_offset", (0, 0))
    render_previews()
    if pan_offset != (0, 0):
        after_canvas.move("after_img", pan_offset[0], pan_offset[1])
        after_preview_origin = (
            after_preview_origin[0] + pan_offset[0],
            after_preview_origin[1] + pan_offset[1],
        )
    update_status()


def set_zoom(new_zoom, record_undo=True):
    global zoom_percent
    if record_undo:
        push_view_undo()
    zoom_percent = max(25, min(400, new_zoom))
    render_previews()
    update_status()


def zoom_in():
    global zoom_tip_shown
    if not zoom_tip_shown:
        messagebox.showinfo("Zoom", "Tip: Right-click and drag to move the photo.")
        zoom_tip_shown = True
    set_zoom(zoom_percent + 25, record_undo=True)


def zoom_out():
    global zoom_tip_shown
    if not zoom_tip_shown:
        messagebox.showinfo("Zoom", "Tip: Right-click and drag to move the photo.")
        zoom_tip_shown = True
    set_zoom(zoom_percent - 25, record_undo=True)


def upload_image():
    global original_bgr, source_bgr, cached_fg_cutout, cached_fg_params, denoise_controls_enabled
    global zoom_percent, pan_offset, pan_start
    path = askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.png *.jpeg")],
    )
    if not path:
        return

    img = cv2.imread(path)
    if img is None:
        return

    source_bgr = img.copy()
    original_bgr = img
    cached_fg_cutout = None
    cached_fg_params = None
    global resize_target
    resize_target = None
    zoom_percent = 100
    pan_offset = (0, 0)
    pan_start = None
    h, w = original_bgr.shape[:2]
    update_resize_fields(w, h, 100)
    process_image()
    set_denoise_controls_enabled(True)
    set_controls_enabled(True)
    update_status()


def toggle_remove_bg():
    global remove_bg, blend_active, cached_fg_cutout, cached_fg_params
    remove_bg = not remove_bg
    if not remove_bg:
        blend_active = False
        cached_fg_cutout = None
        cached_fg_params = None
    remove_bg_button.configure(
        text="BG: ON" if remove_bg else "BG: OFF"
    )
    process_image()
    update_blend_controls()

def choose_background():
    global custom_bg_pil
    path = askopenfilename(
        title="Select a background image",
        filetypes=[("Image files", "*.jpg *.png *.jpeg")],
    )
    if not path:
        return
    try:
        custom_bg_pil = Image.open(path).convert("RGBA")
    except Exception:
        return
    process_image()


def refine_alpha_edges(rgba_img):
    # Reduce edge halos and soften cutout boundary.
    r, g, b, a = rgba_img.split()
    a = a.filter(ImageFilter.MinFilter(3))
    a = a.filter(ImageFilter.GaussianBlur(1.2))
    return Image.merge("RGBA", (r, g, b, a))


def dehalo_foreground(fg_rgba, bg_rgba):
    fg = np.array(fg_rgba)
    bg = np.array(bg_rgba)
    alpha = fg[:, :, 3].astype("float32") / 255.0
    # Replace colors where alpha is very low to avoid light fringes.
    low = alpha < 0.25
    if np.any(low):
        fg[low, :3] = bg[low, :3]
    return Image.fromarray(fg, mode="RGBA")


def transform_layer(img_rgba, target_size, scale, offset):
    if scale <= 0:
        scale = 0.01
    base_w, base_h = target_size
    new_w = max(1, int(base_w * scale))
    new_h = max(1, int(base_h * scale))
    scaled = img_rgba.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGBA", target_size, (0, 0, 0, 0))
    ox, oy = offset
    canvas.paste(scaled, (int(ox), int(oy)), scaled)
    return canvas


def blend_image():
    global processed_pil, custom_bg_pil, remove_bg, blend_active, cached_fg_cutout, cached_fg_params
    if processed_pil is None:
        return

    path = askopenfilename(
        title="Select a background image",
        filetypes=[("Image files", "*.jpg *.png *.jpeg")],
    )
    if not path:
        return

    try:
        custom_bg_pil = Image.open(path).convert("RGBA")
    except Exception:
        return

    remove_bg = True
    blend_active = True
    cached_fg_cutout = None
    cached_fg_params = None
    remove_bg_button.configure(text="BG: ON")
    process_image()
    update_blend_controls()

def update_denoise_mode_button():
    if denoise_mode == "portrait":
        label = "Mode: Port"
    elif denoise_mode == "strong":
        label = "Mode: Strong"
    else:
        label = "Mode: Nat"
    denoise_mode_button.configure(text=label)


def set_denoise_mode(mode):
    global denoise_mode
    denoise_mode = mode
    update_denoise_mode_button()


def show_denoise_mode_menu(_event=None, run_after=False):
    global denoise_mode_menu
    if denoise_mode_menu is None:
        denoise_mode_menu = Menu(root, tearoff=0, bg=THEME["panel"], fg=THEME["text"])
        denoise_mode_menu.add_command(label="Natural", command=lambda: set_denoise_mode("natural"))
        denoise_mode_menu.add_command(label="Portrait", command=lambda: set_denoise_mode("portrait"))
        denoise_mode_menu.add_command(label="Strong", command=lambda: set_denoise_mode("strong"))
    if run_after:
        denoise_mode_menu.entryconfigure(
            0, command=lambda: (set_denoise_mode("natural"), denoise_image())
        )
        denoise_mode_menu.entryconfigure(
            1, command=lambda: (set_denoise_mode("portrait"), denoise_image())
        )
        denoise_mode_menu.entryconfigure(
            2, command=lambda: (set_denoise_mode("strong"), denoise_image())
        )
    else:
        denoise_mode_menu.entryconfigure(
            0, command=lambda: set_denoise_mode("natural")
        )
        denoise_mode_menu.entryconfigure(
            1, command=lambda: set_denoise_mode("portrait")
        )
        denoise_mode_menu.entryconfigure(
            2, command=lambda: set_denoise_mode("strong")
        )
    x = denoise_mode_button.winfo_rootx()
    y = denoise_mode_button.winfo_rooty() + denoise_mode_button.winfo_height()
    denoise_mode_menu.tk_popup(x, y)


def choose_denoise_mode_and_apply():
    global denoise_base_pil
    if processed_pil is None:
        return
    denoise_base_pil = processed_pil.copy()
    show_denoise_mode_menu(run_after=True)

def set_denoise_controls_enabled(enabled):
    global denoise_controls_enabled
    denoise_controls_enabled = enabled
    state = "normal" if enabled else "disabled"
    denoise_scale.configure(state=state)
    denoise_mode_button.configure(state=state)


def denoise_image():
    global processed_pil, original_bgr, denoise_controls_enabled, denoise_base_pil
    if processed_pil is None:
        return
    strength = max(0, min(80, int(denoise_scale.get())))
    if strength <= 0:
        return
    set_denoise_controls_enabled(True)
    if denoise_base_pil is None:
        denoise_base_pil = processed_pil.copy()
    push_undo_from_last()
    alpha = None
    if "A" in denoise_base_pil.getbands():
        alpha = denoise_base_pil.getchannel("A")
    bgr = cv2.cvtColor(np.array(denoise_base_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    h_px, w_px = bgr.shape[:2]
    mp = (w_px * h_px) / 1000000.0

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    residual = gray.astype("float32") - cv2.GaussianBlur(gray, (0, 0), 1.0).astype("float32")
    noise_est = float(np.std(residual))
    noise_factor = np.clip((noise_est - 2.0) / 10.0, 0.55, 1.35)
    if mp < 0.8:
        res_factor = 0.78
    elif mp < 2.0:
        res_factor = 0.9
    elif mp > 8.0:
        res_factor = 1.08
    else:
        res_factor = 1.0
    eff = int(np.clip(strength * noise_factor * res_factor, 6, 70))

    if denoise_mode == "natural":
        h_luma = 4 + int(eff * 0.42)
        h_color = 3 + int(eff * 0.30)
        detail_boost = 0.62
        sharpen_amt = 0.16
        texture_keep = 0.09
    elif denoise_mode == "portrait":
        h_luma = 4 + int(eff * 0.30)
        h_color = 3 + int(eff * 0.22)
        detail_boost = 0.74
        sharpen_amt = 0.14
        texture_keep = 0.10
    else:
        h_luma = 5 + int(eff * 0.38)
        h_color = 4 + int(eff * 0.25)
        detail_boost = 0.68
        sharpen_amt = 0.15
        texture_keep = 0.08

    denoised = cv2.fastNlMeansDenoisingColored(bgr, None, h_luma, h_color, 7, 21)
    if eff >= 40:
        # Smooth chroma only to limit blocky color noise while keeping sharp edges.
        ycrcb = cv2.cvtColor(denoised, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        chroma_sigma = 8 + int((eff - 40) * 0.35)
        cr = cv2.bilateralFilter(cr, d=5, sigmaColor=chroma_sigma, sigmaSpace=10)
        cb = cv2.bilateralFilter(cb, d=5, sigmaColor=chroma_sigma, sigmaSpace=10)
        denoised = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)

    # Detail mask protects high-frequency regions (hair, text, eyelashes, edges).
    edge_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    edge_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = cv2.magnitude(edge_x, edge_y)
    edge_mag = cv2.GaussianBlur(edge_mag, (0, 0), 0.8)
    edge_mask = np.clip(edge_mag / (18.0 + eff * 0.45), 0.0, 1.0).astype("float32")
    edge_mask = cv2.GaussianBlur(edge_mask, (0, 0), 0.9) * detail_boost
    edge_mask_3 = np.dstack([edge_mask, edge_mask, edge_mask])
    flat_mask = np.clip(1.0 - (edge_mask * 1.6), 0.0, 1.0).astype("float32")

    blended = (
        denoised.astype("float32") * (1.0 - edge_mask_3)
        + bgr.astype("float32") * edge_mask_3
    )
    blended = np.clip(blended, 0, 255).astype("uint8")
    # Keep a small amount of source texture to avoid plastic/over-smoothed skin and surfaces.
    blended = cv2.addWeighted(blended, 1.0 - texture_keep, bgr, texture_keep, 0)
    if mp < 1.6 or eff >= 36:
        # Anti-pixel pass on flat regions only (walls, sky, skin), preserving edges.
        ycrcb = cv2.cvtColor(blended, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_smooth = cv2.bilateralFilter(y, d=5, sigmaColor=18, sigmaSpace=8)
        blend_w = 0.34 if denoise_mode == "strong" else 0.28
        fw = (flat_mask * blend_w).astype("float32")
        y_mix = y.astype("float32") * (1.0 - fw) + y_smooth.astype("float32") * fw
        y = np.clip(y_mix, 0, 255).astype("uint8")
        blended = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)

    if strength >= 12:
        # Small unsharp mask on luminance to keep output clear, not blurry.
        ycrcb = cv2.cvtColor(blended, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        blur_y = cv2.GaussianBlur(y, (0, 0), 0.95)
        extra = 0.03 if mp < 1.2 else 0.0
        amount = min(0.22, sharpen_amt + extra)
        y = cv2.addWeighted(y, 1.0 + amount, blur_y, -amount, 0)
        blended = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2BGR)
    if eff >= 52:
        # High-strength safety pass: reduce blocky artifacts without softening edges too much.
        smooth = cv2.bilateralFilter(blended, d=5, sigmaColor=18, sigmaSpace=12)
        if denoise_mode == "strong":
            blended = cv2.addWeighted(smooth, 0.65, blended, 0.35, 0)
        elif denoise_mode == "portrait":
            blended = cv2.addWeighted(smooth, 0.55, blended, 0.45, 0)
        else:
            blended = cv2.addWeighted(smooth, 0.50, blended, 0.50, 0)

    # Keep final tone natural; avoid aggressive local-contrast boosts.

    rgb_out = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    if alpha is not None:
        processed_pil = Image.fromarray(rgb_out).convert("RGBA")
        processed_pil.putalpha(alpha)
    else:
        processed_pil = Image.fromarray(rgb_out)
    render_previews()
    update_status()
    original_bgr = cv2.cvtColor(np.array(processed_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    commit_state()


def reset_all():
    global remove_bg
    global custom_bg_pil
    global resize_target
    global original_bgr, source_bgr, processed_pil
    global zoom_percent, pan_offset, pan_start
    global resize_drag_start, resize_zoom_start
    global undo_stack, redo_stack
    global fg_offset, bg_offset, fg_scale, bg_scale, move_mode, move_drag_start, move_undo_snapshot
    global blend_active, denoise_mode, denoise_controls_enabled, denoise_base_pil
    global cached_fg_cutout, cached_fg_params
    global crop_active, crop_start, crop_rect_id
    remove_bg = False
    custom_bg_pil = None
    blend_active = False
    cached_fg_cutout = None
    cached_fg_params = None
    denoise_controls_enabled = False
    denoise_base_pil = None
    set_denoise_controls_enabled(False)
    if source_bgr is not None:
        original_bgr = source_bgr.copy()
    processed_pil = None
    push_undo_from_last()
    redo_stack.clear()
    remove_bg_button.configure(text="BG: OFF")
    contrast_scale.set(10)
    brightness_scale.set(0)
    saturation_scale.set(100)
    clarity_scale.set(0)
    denoise_scale.set(28)
    denoise_mode = "natural"
    update_denoise_mode_button()
    denoise_controls_enabled = False
    set_denoise_controls_enabled(False)
    blend_scale.set(100)
    resize_target = None
    zoom_percent = 100
    pan_offset = (0, 0)
    pan_start = None
    resize_drag_start = None
    resize_zoom_start = None
    fg_offset = (0, 0)
    bg_offset = (0, 0)
    fg_scale = 1.0
    bg_scale = 1.0
    move_mode = None
    move_drag_start = None
    move_undo_snapshot = None
    crop_active = False
    crop_start = None
    if crop_rect_id is not None:
        after_canvas.delete(crop_rect_id)
        crop_rect_id = None
    if original_bgr is not None:
        h, w = original_bgr.shape[:2]
        update_resize_fields(w, h, 100)
    process_image()
    update_status()
    update_blend_controls()


def save_image():
    if processed_pil is None:
        return

    save_pil = processed_pil

    if remove_bg:
        def_ext = ".png"
        filetypes = [("PNG", "*.png")]
    else:
        def_ext = ".jpg"
        filetypes = [("JPEG", ".jpg"), ("PNG", ".png")]

    save_path = asksaveasfilename(
        defaultextension=def_ext,
        filetypes=filetypes,
    )
    if not save_path:
        return

    base, ext = os.path.splitext(save_path)
    ext = ext.lower()

    if remove_bg:
        if ext != ".png":
            save_path = base + ".png"
        save_pil.save(save_path, format="PNG")
    else:
        if ext == ".png":
            save_pil.save(save_path, format="PNG")
        else:
            if ext not in (".jpg", ".jpeg"):
                save_path = base + ".jpg"
            save_pil.save(save_path, format="JPEG", quality=95)

    root.destroy()


def take_screenshot():
    # Save the processed "After" image directly (not a screen grab)
    if processed_pil is None:
        return

    if remove_bg:
        def_ext = ".png"
        filetypes = [("PNG", "*.png")]
    else:
        def_ext = ".jpg"
        filetypes = [("JPEG", ".jpg"), ("PNG", ".png")]

    save_path = asksaveasfilename(
        defaultextension=def_ext,
        filetypes=filetypes,
    )
    if not save_path:
        return

    base, ext = os.path.splitext(save_path)
    ext = ext.lower()

    if remove_bg:
        if ext != ".png":
            save_path = base + ".png"
        processed_pil.save(save_path, format="PNG")
    else:
        if ext == ".png":
            processed_pil.save(save_path, format="PNG")
        else:
            if ext not in (".jpg", ".jpeg"):
                save_path = base + ".jpg"
        processed_pil.save(save_path, format="JPEG", quality=95)


def toggle_ai_erase():
    global ai_erase_active, zoom_select_active
    if processed_pil is None or original_bgr is None:
        return
    zoom_select_active = False
    ai_erase_active = True
    set_move_mode(None)
    clear_crop_mode()
    root.title("Enhanced Image - AI Erase: select area")
    update_status()


def clear_ai_erase_mode():
    global ai_erase_active
    ai_erase_active = False
    root.title("Enhanced Image")
    update_status()


def preview_to_original_coords(x, y):
    if after_preview_size is None:
        return None
    cx = after_canvas.canvasx(x)
    cy = after_canvas.canvasy(y)
    coords = after_canvas.coords("after_img")
    if coords and len(coords) >= 2:
        ox, oy = coords[0], coords[1]
    else:
        ox, oy = after_preview_origin
    preview_w, preview_h = after_preview_size
    if preview_w == 0 or preview_h == 0:
        return None
    px = cx - ox
    py = cy - oy
    # Clamp to preview bounds to avoid selecting the wrong area.
    if px < 0 or py < 0 or px > preview_w or py > preview_h:
        px = min(max(px, 0), preview_w)
        py = min(max(py, 0), preview_h)
    if processed_pil is not None:
        img_w, img_h = processed_pil.size
    elif original_bgr is not None:
        img_h, img_w = original_bgr.shape[:2]
    else:
        return None
    scale_x = img_w / preview_w
    scale_y = img_h / preview_h
    return int(px * scale_x), int(py * scale_y)


def crop_from_canvas_rect(x0, y0, x1, y1):
    if processed_pil is None:
        return None
    coords = after_canvas.coords("after_img")
    if not coords or len(coords) < 2:
        return None
    ox, oy = coords[0], coords[1]
    if after_preview_size is not None:
        preview_w, preview_h = after_preview_size
    elif after_photo_size is not None:
        preview_w, preview_h = after_photo_size
    else:
        return None
    if preview_w == 0 or preview_h == 0:
        return None
    # Convert to canvas space and clamp to preview bounds.
    cx0 = after_canvas.canvasx(x0)
    cy0 = after_canvas.canvasy(y0)
    cx1 = after_canvas.canvasx(x1)
    cy1 = after_canvas.canvasy(y1)
    px0 = min(max(cx0 - ox, 0), preview_w)
    py0 = min(max(cy0 - oy, 0), preview_h)
    px1 = min(max(cx1 - ox, 0), preview_w)
    py1 = min(max(cy1 - oy, 0), preview_h)
    if px0 > px1:
        px0, px1 = px1, px0
    if py0 > py1:
        py0, py1 = py1, py0
    scale_x = processed_pil.width / preview_w
    scale_y = processed_pil.height / preview_h
    x0i = int(px0 * scale_x)
    y0i = int(py0 * scale_y)
    x1i = int(px1 * scale_x)
    y1i = int(py1 * scale_y)
    return x0i, y0i, x1i, y1i, (px0, py0, px1, py1, ox, oy, preview_w, preview_h)


def smart_inpaint_bgr(bgr, mask):
    img_h, img_w = bgr.shape[:2]

    # Derive adaptive params from the selected area
    x, y, w, h = cv2.boundingRect(mask)
    max_dim = max(w, h) if w > 0 and h > 0 else max(img_w, img_h)

    radius = max(7, int(0.05 * max_dim))
    kernel_size = max(5, int(0.05 * max_dim))
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    soft = cv2.dilate(mask, kernel, iterations=2)
    soft = cv2.GaussianBlur(soft, (kernel_size, kernel_size), 0)
    _, soft = cv2.threshold(soft, 5, 255, cv2.THRESH_BINARY)

    # Two-pass inpaint blends better on textured backgrounds
    telea = cv2.inpaint(bgr, soft, radius, cv2.INPAINT_TELEA)
    ns = cv2.inpaint(telea, soft, radius, cv2.INPAINT_NS)
    return ns


def apply_inpaint(x0, y0, x1, y1):
    global processed_pil, original_bgr
    if processed_pil is None:
        return
    push_undo_from_last()
    img_w, img_h = processed_pil.size
    # Expand the selection a bit to include edges for smoother blending
    pad = 6
    x0 = max(0, min(img_w - 1, x0 - pad))
    x1 = max(0, min(img_w - 1, x1 + pad))
    y0 = max(0, min(img_h - 1, y0 - pad))
    y1 = max(0, min(img_h - 1, y1 + pad))
    if x0 == x1 or y0 == y1:
        return

    mask = np.zeros((img_h, img_w), dtype="uint8")
    cv2.rectangle(mask, (x0, y0), (x1, y1), 255, thickness=-1)

    # Prefer LaMa if available; it gives cleaner results than OpenCV inpaint.
    if SimpleLama is not None:
        global simple_lama
        if simple_lama is None:
            simple_lama = SimpleLama()
        image_pil = processed_pil.convert("RGB")
        mask_pil = Image.fromarray(mask).convert("L")
        result_pil = simple_lama(image_pil, mask_pil)
        result_rgb = np.array(result_pil)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        result_bgr = smart_inpaint_bgr(result_bgr, mask)
        processed_pil = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
    else:
        bgr = cv2.cvtColor(np.array(processed_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        result_bgr = smart_inpaint_bgr(bgr, mask)
        processed_pil = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
    render_previews()
    update_status()
    # Persist the erase so further edits don't bring it back.
    original_bgr = cv2.cvtColor(np.array(processed_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    commit_state()


def on_after_mouse_down(event):
    global drag_start, drag_rect_id, move_drag_start, move_undo_snapshot, crop_start, crop_rect_id
    if move_mode in ("fg", "bg"):
        if remove_bg and custom_bg_pil is not None and processed_pil is not None:
            move_drag_start = (event.x, event.y)
            move_undo_snapshot = processed_pil.copy()
        return
    if crop_active:
        crop_start = (event.x, event.y)
        if crop_rect_id is not None:
            after_canvas.delete(crop_rect_id)
            crop_rect_id = None
        crop_rect_id = after_canvas.create_rectangle(
            event.x,
            event.y,
            event.x,
            event.y,
            outline=THEME["accent_2"],
            width=2,
        )
        return
    if zoom_select_active:
        return
    if not ai_erase_active or processed_pil is None:
        return
    drag_start = (event.x, event.y)
    if drag_rect_id is not None:
        after_canvas.delete(drag_rect_id)
        drag_rect_id = None
    drag_rect_id = after_canvas.create_rectangle(
        event.x,
        event.y,
        event.x,
        event.y,
        outline="red",
        width=2,
    )


def on_after_mouse_drag(event):
    global move_drag_start, fg_offset, bg_offset, suppress_undo
    if move_mode in ("fg", "bg") and move_drag_start is not None:
        if processed_pil is None or after_preview_size is None:
            return
        img_w, img_h = processed_pil.size
        preview_w, preview_h = after_preview_size
        if preview_w == 0 or preview_h == 0:
            return
        dx = event.x - move_drag_start[0]
        dy = event.y - move_drag_start[1]
        scale_x = img_w / preview_w
        scale_y = img_h / preview_h
        dx_img = int(dx * scale_x)
        dy_img = int(dy * scale_y)
        if move_mode == "fg":
            fg_offset = (fg_offset[0] + dx_img, fg_offset[1] + dy_img)
        else:
            bg_offset = (bg_offset[0] + dx_img, bg_offset[1] + dy_img)
        move_drag_start = (event.x, event.y)
        suppress_undo = True
        try:
            process_image()
        finally:
            suppress_undo = False
        return
    if crop_active and crop_start is not None and crop_rect_id is not None:
        after_canvas.coords(crop_rect_id, crop_start[0], crop_start[1], event.x, event.y)
        return
    if zoom_select_active:
        return
    if not ai_erase_active or drag_start is None or drag_rect_id is None:
        return
    after_canvas.coords(drag_rect_id, drag_start[0], drag_start[1], event.x, event.y)


def on_after_mouse_up(event):
    global drag_start, drag_rect_id, move_drag_start, move_undo_snapshot, undo_stack
    global crop_start, crop_rect_id, crop_active, processed_pil, crop_outline_id
    if move_mode in ("fg", "bg") and move_drag_start is not None:
        push_undo_from_last()
        commit_state()
        move_undo_snapshot = None
        move_drag_start = None
        return
    if crop_active and crop_start is not None:
        rect = crop_from_canvas_rect(crop_start[0], crop_start[1], event.x, event.y)
        if rect:
            x0, y0, x1, y1, dbg = rect
            px0, py0, px1, py1, ox, oy, pw, ph = dbg
            # Draw debug outline of mapped crop on the preview image.
            if crop_outline_id is not None:
                after_canvas.delete(crop_outline_id)
            crop_outline_id = after_canvas.create_rectangle(
                ox + px0,
                oy + py0,
                ox + px1,
                oy + py1,
                outline="#00e5ff",
                width=2,
            )
            min_size = 8
            if (x1 - x0) < min_size or (y1 - y0) < min_size:
                clear_crop_mode()
                return
            if messagebox.askokcancel("Crop", "Crop to the selected area?"):
                if processed_pil is not None:
                    push_undo_from_last()
                    processed_pil = processed_pil.crop((x0, y0, x1, y1))
                    render_previews()
                    update_status()
                    commit_state()
        clear_crop_mode()
        return
    if zoom_select_active:
        return
    if not ai_erase_active or drag_start is None:
        return

    start = preview_to_original_coords(drag_start[0], drag_start[1])
    end = preview_to_original_coords(event.x, event.y)
    if start and end:
        x0, y0 = start
        x1, y1 = end
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        # Ensure a minimum selection size
        min_size = 8
        if (x1 - x0) < min_size:
            mid = (x0 + x1) // 2
            x0 = max(0, mid - min_size // 2)
            x1 = mid + min_size // 2
        if (y1 - y0) < min_size:
            mid = (y0 + y1) // 2
            y0 = max(0, mid - min_size // 2)
            y1 = mid + min_size // 2
        if messagebox.askokcancel("AI Erase", "Delete the selected area?"):
            apply_inpaint(x0, y0, x1, y1)

    if drag_rect_id is not None:
        after_canvas.delete(drag_rect_id)
        drag_rect_id = None
    drag_start = None
    clear_ai_erase_mode()


def on_after_resize_start(event):
    global resize_drag_start, resize_zoom_start, resize_tip_shown
    if ai_erase_active or zoom_select_active or crop_active or move_mode in ("fg", "bg") or processed_pil is None:
        return
    if not resize_tip_shown:
        messagebox.showinfo("Resize", "Tip: Left-click and drag to resize (zoom) the photo.")
        resize_tip_shown = True
    resize_drag_start = (event.x, event.y)
    resize_zoom_start = zoom_percent


def on_after_resize_drag(event):
    if ai_erase_active or zoom_select_active or crop_active or move_mode in ("fg", "bg") or resize_drag_start is None or resize_zoom_start is None:
        return
    dy = resize_drag_start[1] - event.y
    new_zoom = resize_zoom_start + int(dy / 2)
    set_zoom(new_zoom, record_undo=False)


def on_after_resize_end(_event):
    global resize_drag_start, resize_zoom_start
    if move_mode in ("fg", "bg") or crop_active:
        return
    if resize_zoom_start is not None and zoom_percent != resize_zoom_start:
        push_view_undo()
    resize_drag_start = None
    resize_zoom_start = None


def on_after_pan_start(event):
    global pan_start
    if ai_erase_active or zoom_select_active or crop_active or move_mode in ("fg", "bg"):
        return
    pan_start = (event.x, event.y)


def on_after_pan_drag(event):
    global pan_start, pan_offset, after_preview_origin
    if ai_erase_active or zoom_select_active or crop_active or move_mode in ("fg", "bg") or pan_start is None:
        return
    dx = event.x - pan_start[0]
    dy = event.y - pan_start[1]
    if dx == 0 and dy == 0:
        return
    after_canvas.move("after_img", dx, dy)
    pan_start = (event.x, event.y)
    pan_offset = (pan_offset[0] + dx, pan_offset[1] + dy)
    after_preview_origin = (
        after_preview_origin[0] + dx,
        after_preview_origin[1] + dy,
    )


def on_after_pan_end(_event):
    if ai_erase_active or zoom_select_active or crop_active or move_mode in ("fg", "bg"):
        return
    if pan_offset != (0, 0):
        push_view_undo()


def on_after_mouse_wheel(event):
    global fg_scale, bg_scale
    if move_mode not in ("fg", "bg"):
        return
    if not (remove_bg and custom_bg_pil is not None and processed_pil is not None):
        return
    delta = 1 if event.delta > 0 else -1
    step = 0.05
    if move_mode == "fg":
        fg_scale = max(0.2, min(3.0, fg_scale + delta * step))
    else:
        bg_scale = max(0.2, min(3.0, bg_scale + delta * step))
    process_image()


def update_status():
    if move_mode == "fg":
        mode = "Move FG"
    elif move_mode == "bg":
        mode = "Move BG"
    elif ai_erase_active:
        mode = "AI Erase"
    elif zoom_select_active:
        mode = "Zoom Select"
    elif crop_active:
        mode = "Crop"
    else:
        mode = "Normal"
    if processed_pil is not None:
        w, h = processed_pil.size
    elif original_bgr is not None:
        h, w = original_bgr.shape[:2]
    else:
        w, h = 0, 0
    status_text.set(f"Zoom: {zoom_percent}%   |   Resolution: {w} x {h}   |   Mode: {mode}")


def toggle_zoom_select():
    global zoom_select_active, ai_erase_active
    if processed_pil is None:
        return
    ai_erase_active = False
    zoom_select_active = True
    set_move_mode(None)
    clear_crop_mode()
    root.title("Enhanced Image - Zoom to Selection: drag a rectangle")
    update_status()


def clear_zoom_select_mode():
    global zoom_select_active, zoom_select_start, zoom_select_rect_id
    zoom_select_active = False
    zoom_select_start = None
    if zoom_select_rect_id is not None:
        after_canvas.delete(zoom_select_rect_id)
        zoom_select_rect_id = None
    root.title("Enhanced Image")
    update_status()


def toggle_crop():
    global crop_active, ai_erase_active, zoom_select_active
    if processed_pil is None:
        return
    ai_erase_active = False
    zoom_select_active = False
    set_move_mode(None)
    crop_active = not crop_active
    if crop_active:
        root.title("Enhanced Image - Crop: drag a rectangle")
    else:
        root.title("Enhanced Image")
    update_status()


def clear_crop_mode():
    global crop_active, crop_start, crop_rect_id, crop_outline_id
    crop_active = False
    crop_start = None
    if crop_rect_id is not None:
        after_canvas.delete(crop_rect_id)
        crop_rect_id = None
    if crop_outline_id is not None:
        after_canvas.delete(crop_outline_id)
        crop_outline_id = None


def set_move_mode(mode):
    global move_mode, move_drag_start
    move_mode = mode
    move_drag_start = None
    if mode is None:
        root.title("Enhanced Image")
    else:
        root.title(f"Enhanced Image - Move {mode.upper()}: drag to move, wheel to scale")
    update_status()


def toggle_move_fg():
    global ai_erase_active, zoom_select_active
    if processed_pil is None:
        return
    ai_erase_active = False
    zoom_select_active = False
    set_move_mode(None if move_mode == "fg" else "fg")


def toggle_move_bg():
    global ai_erase_active, zoom_select_active
    if processed_pil is None:
        return
    ai_erase_active = False
    zoom_select_active = False
    set_move_mode(None if move_mode == "bg" else "bg")


def on_zoom_select_down(event):
    global zoom_select_start, zoom_select_rect_id
    if not zoom_select_active or processed_pil is None:
        return
    zoom_select_start = (event.x, event.y)
    if zoom_select_rect_id is not None:
        after_canvas.delete(zoom_select_rect_id)
        zoom_select_rect_id = None
    zoom_select_rect_id = after_canvas.create_rectangle(
        event.x,
        event.y,
        event.x,
        event.y,
        outline=THEME["accent"],
        width=2,
    )


def on_zoom_select_drag(event):
    if not zoom_select_active or zoom_select_start is None or zoom_select_rect_id is None:
        return
    after_canvas.coords(zoom_select_rect_id, zoom_select_start[0], zoom_select_start[1], event.x, event.y)


def on_zoom_select_up(event):
    global zoom_select_start, zoom_select_rect_id, pan_offset, after_preview_origin
    if not zoom_select_active or zoom_select_start is None or processed_pil is None:
        return
    start = preview_to_original_coords(zoom_select_start[0], zoom_select_start[1])
    end = preview_to_original_coords(event.x, event.y)
    if start and end:
        x0, y0 = start
        x1, y1 = end
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        sel_w = max(1, x1 - x0)
        sel_h = max(1, y1 - y0)

        # Compute zoom so the selection fits the preview while preserving aspect ratio.
        base_preview = fit_preview(processed_pil, 100, cap_size=True)
        base_scale_x = base_preview.width / processed_pil.width
        base_scale_y = base_preview.height / processed_pil.height
        sel_preview_w = sel_w * base_scale_x
        sel_preview_h = sel_h * base_scale_y
        if sel_preview_w > 0 and sel_preview_h > 0:
            target_scale = min(MAX_PREVIEW_W / sel_preview_w, MAX_PREVIEW_H / sel_preview_h)
            new_zoom = int(max(25, min(400, 100 * target_scale)))
            set_zoom(new_zoom, record_undo=True)

            # Center the selected region in the preview.
            after_preview = fit_preview(processed_pil, zoom_percent, cap_size=True)
            scale_x = after_preview.width / processed_pil.width
            scale_y = after_preview.height / processed_pil.height
            sel_cx = (x0 + x1) / 2.0
            sel_cy = (y0 + y1) / 2.0
            sel_px = sel_cx * scale_x
            sel_py = sel_cy * scale_y
            ax = (MAX_PREVIEW_W - after_preview.width) // 2
            ay = (MAX_PREVIEW_H - after_preview.height) // 2
            desired_px = MAX_PREVIEW_W / 2.0
            desired_py = MAX_PREVIEW_H / 2.0
            dx = desired_px - (ax + sel_px)
            dy = desired_py - (ay + sel_py)
            after_canvas.move("after_img", dx, dy)
            pan_offset = (dx, dy)
            after_preview_origin = (ax + dx, ay + dy)

    if zoom_select_rect_id is not None:
        after_canvas.delete(zoom_select_rect_id)
        zoom_select_rect_id = None
    zoom_select_start = None
    clear_zoom_select_mode()


def undo_last_action():
    global processed_pil, undo_stack, redo_stack
    if not undo_stack:
        return
    state = undo_stack.pop()
    if state.get("kind") == "full":
        redo_stack.append(capture_state())
        restore_state(state)
    elif "image" in state:
        redo_stack.append({"image": processed_pil.copy()} if processed_pil is not None else {"image": None})
        processed_pil = state["image"]
        render_previews()
        update_status()
        commit_state()
    else:
        redo_stack.append({"zoom": zoom_percent, "pan_offset": pan_offset})
        apply_view_state(state)


def redo_last_action():
    global processed_pil, undo_stack, redo_stack
    if not redo_stack:
        return
    state = redo_stack.pop()
    if state.get("kind") == "full":
        undo_stack.append(capture_state())
        restore_state(state)
    elif "image" in state:
        if processed_pil is not None:
            undo_stack.append({"image": processed_pil.copy()})
        processed_pil = state["image"]
        render_previews()
        update_status()
        commit_state()
    else:
        undo_stack.append({"zoom": zoom_percent, "pan_offset": pan_offset})
        apply_view_state(state)


def fit_screen():
    global zoom_percent, pan_offset, pan_start
    push_view_undo()
    zoom_percent = 100
    pan_offset = (0, 0)
    pan_start = None
    fg_offset = (0, 0)
    bg_offset = (0, 0)
    fg_scale = 1.0
    bg_scale = 1.0
    move_mode = None
    move_drag_start = None
    move_undo_snapshot = None
    render_previews()
    update_status()


def rotate_image():
    global processed_pil, original_bgr, resize_target, denoise_base_pil
    if processed_pil is None:
        return
    push_undo_from_last()
    processed_pil = processed_pil.transpose(Image.ROTATE_270)
    original_bgr = cv2.cvtColor(np.array(processed_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    if resize_target is not None:
        rw, rh = resize_target
        resize_target = (rh, rw)
    denoise_base_pil = None
    h, w = original_bgr.shape[:2]
    update_resize_fields(w, h, 100)
    render_previews()
    update_status()
    commit_state()


# Wire events
update_denoise_mode_button()
set_denoise_controls_enabled(False)
contrast_scale.configure(command=on_slider_change)
brightness_scale.configure(command=on_slider_change)
saturation_scale.configure(command=on_slider_change)
clarity_scale.configure(command=on_slider_change)
blend_scale.configure(command=on_slider_change)
denoise_scale.configure(command=on_denoise_change)
contrast_scale.bind("<ButtonPress-1>", on_slider_press)
brightness_scale.bind("<ButtonPress-1>", on_slider_press)
saturation_scale.bind("<ButtonPress-1>", on_slider_press)
clarity_scale.bind("<ButtonPress-1>", on_slider_press)
blend_scale.bind("<ButtonPress-1>", on_slider_press)
denoise_scale.bind("<ButtonPress-1>", on_slider_press)
contrast_scale.bind("<ButtonRelease-1>", on_slider_release)
brightness_scale.bind("<ButtonRelease-1>", on_slider_release)
saturation_scale.bind("<ButtonRelease-1>", on_slider_release)
clarity_scale.bind("<ButtonRelease-1>", on_slider_release)
blend_scale.bind("<ButtonRelease-1>", on_slider_release)
denoise_scale.bind("<ButtonRelease-1>", on_denoise_release)
upload_button.configure(command=upload_image)
remove_bg_button.configure(command=toggle_remove_bg)
bg_choice_button.configure(command=choose_background)
blend_button.configure(command=blend_image)
reset_button.configure(command=reset_all)
screen_button.configure(command=take_screenshot)
save_button.configure(command=save_image)
ai_erase_button.configure(command=toggle_ai_erase)
zoom_select_button.configure(command=toggle_zoom_select)
undo_button.configure(command=undo_last_action)
redo_button.configure(command=redo_last_action)
fit_screen_button.configure(command=fit_screen)
apply_wh_button.configure(command=apply_resize_wh)
apply_percent_button.configure(command=apply_resize_percent)
percent_scale.configure(command=on_percent_change)
percent_scale.bind("<ButtonPress-1>", on_slider_press)
percent_scale.bind("<ButtonRelease-1>", on_percent_release)
zoom_in_button.configure(command=zoom_in)
zoom_out_button.configure(command=zoom_out)
rotate_button.configure(command=rotate_image)
crop_button.configure(command=toggle_crop)
move_fg_button.configure(command=toggle_move_fg)
move_bg_button.configure(command=toggle_move_bg)
denoise_button.configure(command=choose_denoise_mode_and_apply)
denoise_mode_button.configure(command=show_denoise_mode_menu)

after_canvas.bind("<ButtonPress-1>", on_after_mouse_down)
after_canvas.bind("<B1-Motion>", on_after_mouse_drag)
after_canvas.bind("<ButtonRelease-1>", on_after_mouse_up)
after_canvas.bind("<ButtonPress-1>", on_after_resize_start, add="+")
after_canvas.bind("<B1-Motion>", on_after_resize_drag, add="+")
after_canvas.bind("<ButtonRelease-1>", on_after_resize_end, add="+")
after_canvas.bind("<ButtonPress-1>", on_zoom_select_down, add="+")
after_canvas.bind("<B1-Motion>", on_zoom_select_drag, add="+")
after_canvas.bind("<ButtonRelease-1>", on_zoom_select_up, add="+")
after_canvas.bind("<ButtonPress-3>", on_after_pan_start)
after_canvas.bind("<B3-Motion>", on_after_pan_drag)
after_canvas.bind("<ButtonRelease-3>", on_after_pan_end)
after_canvas.bind("<MouseWheel>", on_after_mouse_wheel)
right_panel.bind("<Configure>", refresh_preview_layout)

# Status bar
status_text.set("Zoom: 100%   |   Resolution: 0 x 0   |   Mode: Normal")
status_bar = Label(root, textvariable=status_text, bg=THEME["panel"], fg=THEME["muted"], anchor="w", padx=10)
status_bar.pack(side="bottom", fill="x")
refresh_preview_layout()
root.mainloop()
