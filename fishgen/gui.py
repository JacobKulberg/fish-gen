"""Desktop GUI: pick a fish type, tweak every gene, watch it swim, export it.

Run with ``py gui.py`` from the repo, ``fishgen gui`` or ``python -m fishgen``.
Rendering happens on a worker thread so the sliders stay responsive.
"""

from __future__ import annotations

import colorsys
import copy
import os
import subprocess
import sys
import threading
import time
import queue
import random
import tkinter as tk
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .animation import SHOWCASE_ORDER, FishSprite, generate
from .export import FORMATS, export_all
from .genome import (ALL_FIN_SHAPES, ALL_PATTERNS, ALL_TAIL_SHAPES, ARCHETYPES, EYE_STYLES, KINDS, LOBE_SHAPES,
                     PECTORAL_SHAPES, WING_SHAPES, FishGenome)
from .kinds import kind_from_label, kind_label
from .palette import ACCENT, ACCENT2, BASE, FIN, build_palette
from .species import RANDOM, kind_of, make_genome, species_for

ANY = "Any"
SHOWCASE = "Showcase (full cycle)"
BACKGROUNDS = ("Water", "Dark", "Light", "Checker")
FORMAT_LABELS = {
    "gif": "Animated GIFs",
    "sheet": "Sprite sheet + JSON",
    "godot": "Godot 4 SpriteFrames",
    "frames": "Individual PNG frames",
}

# Gene controls, grouped by tab and section. A section is (title, kinds it applies to
# or None for every kind, controls); sections for other kinds are hidden, and so is a
# tab with nothing left to show. Each control is (label, genome path, type, extra)
# where type is "float" (extra = (min, max)), "int" (extra = (min, max)),
# "choice" (extra = options), "bool", or "color" (an optional #rrggbb override).
SWIMMERS = ("fish", "shark", "whale", "ray", "squid", "octopus")
FINNED = ("fish", "shark", "whale")
GENE_TABS: list[tuple[str, list[tuple[str, tuple | None, list[tuple]]]]] = [
    ("Body", [
        ("Size", None, [
            ("Animal size", "size", "float", (0.3, 3.5)),
        ]),
        ("Shape", None, [
            ("Body height", "height", "float", (0.06, 0.95)),
            ("Thickness", "thickness", "float", (0.3, 2.8)),
            ("Tallest point", "peak", "float", (0.2, 0.6)),
            ("Nose pointiness", "nose", "float", (0.2, 2.0)),
            ("Flat head", "flat_head", "float", (0.0, 1.0)),
            ("Rear taper", "taper", "float", (0.4, 3.0)),
            ("Tail stem", "peduncle", "float", (0.08, 0.7)),
            ("Back arch", "back_arch", "float", (0.35, 0.75)),
        ]),
        ("Shark head", ("shark",), [
            ("Gill slits", "gill_slits", "int", (0, 7)),
            ("Hammer head", "hammer", "float", (0.0, 0.25)),
        ]),
        ("Dolphin head", ("whale",), [
            ("Beak", "beak", "float", (0.0, 0.15)),
            ("Melon (forehead)", "melon", "float", (0.0, 0.08)),
            ("Neck", "neck", "float", (0.0, 0.4)),
            ("Throat grooves", "pleats", "bool", None),
        ]),
        ("Mouth", ("fish", "shark", "whale"), [
            ("Mouth length", "mouth", "float", (0.0, 0.2)),
            ("Mouth height", "mouth_height", "float", (-0.9, 0.5)),
            ("Teeth", "teeth", "bool", None),
        ]),
        ("Head lobes", ("ray",), [
            ("Lobes", "lobes", "choice", LOBE_SHAPES),
            ("Lobe length", "lobe_length", "float", (0.0, 0.15)),
        ]),
        ("Puffing up", ("fish",), [
            ("Puffed size", "puff", "float", (0.0, 1.2)),
            ("Spikes", "puff_spines", "int", (0, 200)),
        ]),
        ("Eye", SWIMMERS, [
            ("Position", "eye_pos", "float", (0.02, 0.9)),
            ("Height", "eye_height", "float", (-0.2, 0.8)),
            ("Size", "eye_size", "float", (0.1, 0.4)),
            ("Style", "eye_style", "choice", EYE_STYLES),
        ]),
        ("Swimming", SWIMMERS, [
            ("Tail swing", "swim_amplitude", "float", (0.0, 1.0)),
            ("Body waves", "wave_count", "float", (0.3, 1.6)),
            ("Mantle pulse", "pulse", "float", (0.0, 0.25)),
        ]),
        ("Motion & camera", None, [
            ("Turn bend", "turn_lag", "float", (0.0, 1.2)),
            ("Extra camera tilt", "camera_tilt", "float", (0.0, 0.8)),
        ]),
    ]),
    ("Fins", [
        ("Tail fin", FINNED, [
            ("Shape", "tail_shape", "choice", ALL_TAIL_SHAPES),
            ("Length", "tail_length", "float", (0.1, 0.6)),
            ("Spread", "tail_spread", "float", (0.5, 2.0)),
        ]),
        ("Dorsal fin (top)", ("fish", "shark", "whale", "ray"), [
            ("Show", "dorsal.present", "bool", None),
            ("Shape", "dorsal.shape", "choice", ALL_FIN_SHAPES),
            ("Starts at", "dorsal.start", "float", (0.1, 0.9)),
            ("Ends at", "dorsal.end", "float", (0.15, 1.0)),
            ("Height", "dorsal.height", "float", (0.02, 0.5)),
            ("Sweep back", "dorsal.sweep", "float", (0.0, 1.0)),
        ]),
        ("Second dorsal fin", ("shark",), [
            ("Show", "dorsal2.present", "bool", None),
            ("Starts at", "dorsal2.start", "float", (0.4, 0.95)),
            ("Height", "dorsal2.height", "float", (0.02, 0.2)),
        ]),
        ("Anal fin (bottom)", ("fish", "shark"), [
            ("Show", "anal.present", "bool", None),
            ("Shape", "anal.shape", "choice", ALL_FIN_SHAPES),
            ("Starts at", "anal.start", "float", (0.1, 0.9)),
            ("Ends at", "anal.end", "float", (0.15, 1.0)),
            ("Height", "anal.height", "float", (0.02, 0.5)),
            ("Sweep back", "anal.sweep", "float", (0.0, 1.0)),
        ]),
        ("Side fins", FINNED, [
            ("Pectoral size", "pectoral_size", "float", (0.05, 0.6)),
            ("Pectoral shape", "pectoral_shape", "choice", PECTORAL_SHAPES),
            ("Pectoral angle", "pectoral_droop", "float", (0.2, 1.6)),
            ("Fin flutter", "pectoral_flap", "float", (0.0, 1.0)),
            ("Pelvic fins", "pelvic.present", "bool", None),
            ("Pelvic position", "pelvic.start", "float", (0.2, 0.6)),
            ("Pelvic size", "pelvic.height", "float", (0.03, 0.25)),
        ]),
    ]),
    ("Wings & arms", [
        ("Wings", ("ray", "squid", "octopus"), [
            ("Show", "wings.present", "bool", None),
            ("Shape", "wings.shape", "choice", WING_SHAPES),
            ("Starts at", "wings.start", "float", (0.0, 0.8)),
            ("Ends at", "wings.end", "float", (0.05, 1.0)),
            ("Span", "wings.span", "float", (0.02, 0.8)),
            ("Tip sweep", "wings.sweep", "float", (0.0, 1.5)),
            ("Flap", "wings.flap", "float", (0.0, 1.0)),
            ("Ripple", "wings.wave", "float", (0.0, 2.5)),
        ]),
        ("Arms", ("squid", "octopus"), [
            ("Arms", "arms.count", "int", (0, 10)),
            ("Arm length", "arms.length", "float", (0.1, 2.0)),
            ("Thickness", "arms.thickness", "float", (0.01, 0.1)),
            ("Tentacles", "arms.tentacles", "int", (0, 2)),
            ("Tentacle length", "arms.tentacle_length", "float", (0.2, 2.0)),
            ("Spread", "arms.spread", "float", (0.0, 1.2)),
            ("Writhe", "arms.curl", "float", (0.0, 1.0)),
        ]),
        ("Spines", ("urchin",), [
            ("Spines", "spines.count", "int", (0, 600)),
            ("Length", "spines.length", "float", (0.05, 1.2)),
            ("Sway", "spines.sway", "float", (0.0, 0.4)),
        ]),
    ]),
    ("Colours", [
        ("Body", None, [
            ("Hue", "colors.hue", "float", (0.0, 1.0)),
            ("Saturation", "colors.saturation", "float", (0.0, 1.0)),
            ("Brightness", "colors.value", "float", (0.1, 1.0)),
            ("Pale belly", "colors.belly_lightness", "float", (0.0, 1.0)),
        ]),
        ("Fins & spines", ("fish", "shark", "whale", "ray", "urchin"), [
            ("Fin colour", "colors.fin_color", "color", FIN),
            ("Hue shift (auto)", "colors.fin_hue_shift", "float", (-0.5, 0.5)),
        ]),
        ("Pattern", None, [
            ("Pattern", "colors.pattern", "choice", ALL_PATTERNS),
            ("Pattern colour", "colors.accent_color", "color", ACCENT),
            ("Second colour", "colors.accent2_color", "color", ACCENT2),
            ("Stripes on fins", "colors.pattern_on_fins", "bool", None),
            ("Tail in body colour", "colors.tail_body_color", "bool", None),
            ("Hue shift (auto)", "colors.accent_hue_shift", "float", (-0.5, 0.5)),
            ("Density", "colors.pattern_scale", "float", (0.0, 1.0)),
            ("Offset", "colors.pattern_offset", "float", (0.0, 1.0)),
            ("Fin tips", "colors.fin_tips", "float", (0.0, 0.6)),
        ]),
    ]),
]


def _get(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _set(obj, path: str, value):
    *parents, last = path.split(".")
    for part in parents:
        obj = getattr(obj, part)
    setattr(obj, last, value)


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(rgb)


def _open_folder(path: Path):
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


class ScrollFrame(ttk.Frame):
    """A vertically scrolling container; put widgets in ``.inner``."""

    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, padding=(10, 6, 14, 10))
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        self.bind("<Enter>", lambda e: self.bind_all("<MouseWheel>", self._wheel))
        self.bind("<Leave>", lambda e: self.unbind_all("<MouseWheel>"))

    def _wheel(self, event):
        if self.inner.winfo_height() > self.canvas.winfo_height():
            self.canvas.yview_scroll(int(-event.delta / 120) or (-1 if event.delta > 0 else 1), "units")


class FishStudio:
    def __init__(self, root: tk.Tk, seed: int | None = None, species: str = RANDOM, out: Path = Path("out"),
                 kind: str = "fish"):
        self.root = root
        root.title("fishgen studio")
        root.geometry("1180x760")
        root.minsize(900, 600)

        self.gene_vars: dict[str, tk.Variable] = {}
        self.swatches: dict[str, tuple[tk.Label, int]] = {}
        self.base_genome: FishGenome | None = None
        self.sprite: FishSprite | None = None
        self._loading = False
        self._batch = False
        self._render_after: str | None = None
        self._job = 0
        self._shown_job = 0
        self._pending = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._results: queue.Queue = queue.Queue()
        self._frames: list[ImageTk.PhotoImage] = []
        self._frame_i = 0
        self._name_is_auto = True

        # Animal: kind first, then a species of that kind
        self.kind = tk.StringVar(value=kind_label(kind if species == RANDOM else kind_of(species)))
        self.species = tk.StringVar(value=species)
        self.archetype = tk.StringVar(value=ANY)
        self.seed = tk.IntVar(value=seed if seed is not None else random.randrange(1_000_000))
        # Sprite settings
        self.width = tk.IntVar(value=32)
        self.height = tk.IntVar(value=32)
        self.swim_frames = tk.IntVar(value=8)
        self.turn_frames = tk.IntVar(value=10)
        self.fps = tk.DoubleVar(value=10.0)
        self.pitch = tk.DoubleVar(value=0.25)
        self.scale_by_size = tk.BooleanVar(value=True)
        # Export
        self.name = tk.StringVar()
        self.out_dir = tk.StringVar(value=str(out.resolve()))
        self.formats = {f: tk.BooleanVar(value=True) for f in FORMATS}
        self.gif_scale = tk.IntVar(value=4)
        self.frame_scale = tk.IntVar(value=1)
        self.gif_bg_on = tk.BooleanVar(value=False)
        self.gif_bg = tk.StringVar(value="#1d3557")
        self.padding = tk.IntVar(value=0)
        self.godot_dir = tk.StringVar(value="res://")
        # Preview
        self.anim = tk.StringVar(value=SHOWCASE)
        self.zoom = tk.StringVar(value="Fit")
        self.background = tk.StringVar(value="Water")
        self.playing = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="")

        self._build()

        self.kind.trace_add("write", lambda *_: self._kind_changed())
        for v in (self.species, self.archetype, self.seed):
            v.trace_add("write", lambda *_: self._type_changed())
        for v in (self.width, self.height, self.swim_frames, self.turn_frames, self.fps, self.pitch, self.scale_by_size):
            v.trace_add("write", lambda *_: self.request_render())
        for v in (self.anim, self.zoom, self.background):
            v.trace_add("write", lambda *_: self.rebuild_frames())
        self.name.trace_add("write", lambda *_: self._name_edited())

        threading.Thread(target=self._worker, daemon=True).start()
        self.load_type()
        self.root.after(50, self._poll)
        self.root.after(100, self._tick)

    # ------------------------------------------------------------------ layout
    def _build(self):
        style = ttk.Style()
        style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Big.TButton", padding=(14, 6))

        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True)
        left = ttk.Frame(paned, width=430)
        right = ttk.Frame(paned)
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        self._build_type(left)
        nb = self.notebook = ttk.Notebook(left)
        nb.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        # (tab, [(section frame, kinds or None)]) so sections can follow the chosen kind.
        self.gene_tabs: list[tuple[ttk.Frame, list[tuple[ttk.Frame, tuple | None]]]] = []
        for title, sections in GENE_TABS:
            sf = ScrollFrame(nb)
            nb.add(sf, text=title)
            self.gene_tabs.append((sf, self._build_gene_tab(sf.inner, sections)))
        sf = ScrollFrame(nb)
        nb.add(sf, text="Sprite")
        self._build_sprite_tab(sf.inner)
        sf = ScrollFrame(nb)
        nb.add(sf, text="Export")
        self._build_export_tab(sf.inner)

        self._build_preview(right)

    def _build_type(self, parent):
        box = ttk.LabelFrame(parent, text=" Animal ", padding=10)
        box.pack(fill="x", padx=8, pady=8)
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text="Kind").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Combobox(box, textvariable=self.kind, values=[kind_label(k) for k in KINDS], state="readonly",
                     width=18).grid(row=0, column=1, columnspan=2, sticky="ew", pady=3)

        ttk.Label(box, text="Species").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=3)
        self.species_box = ttk.Combobox(box, textvariable=self.species, values=species_for(self._kind()),
                                        state="readonly", width=18)
        self.species_box.grid(row=1, column=1, columnspan=2, sticky="ew", pady=3)

        ttk.Label(box, text="Body type").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=3)
        self.arch_box = ttk.Combobox(box, textvariable=self.archetype, values=(ANY, *ARCHETYPES),
                                     state="readonly", width=18)
        self.arch_box.grid(row=2, column=1, columnspan=2, sticky="ew", pady=3)

        ttk.Label(box, text="Seed").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Spinbox(box, textvariable=self.seed, from_=0, to=999_999_999, width=12) \
            .grid(row=3, column=1, sticky="ew", pady=3)
        ttk.Button(box, text="Random seed", command=self.random_seed).grid(row=3, column=2, sticky="ew", padx=(6, 0))

        btns = ttk.Frame(box)
        btns.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        ttk.Button(btns, text="Surprise me", command=self.surprise).pack(side="left")
        ttk.Button(btns, text="Undo my tweaks", command=self.load_type).pack(side="left", padx=6)
        ttk.Label(box, text="Kind + species + seed = same animal. Tweaks below go on top.",
                  foreground="#666").grid(row=5, column=0, columnspan=3, sticky="w", pady=(6, 0))

    def _section(self, parent, row: int, title: str) -> int:
        ttk.Label(parent, text=title, style="Section.TLabel").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(12 if row else 2, 4))
        return row + 1

    def _slider(self, parent, row: int, label: str, var: tk.Variable, lo: float, hi: float,
                fmt: str = "{:.2f}", step: float | None = None):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(8, 10), pady=2)
        value = ttk.Label(parent, width=6, anchor="e")
        value.grid(row=row, column=2, sticky="e")

        def moved(text):
            if step:
                v = round(float(text) / step) * step
                if var.get() != v:
                    var.set(v)

        scale = ttk.Scale(parent, from_=lo, to=hi, variable=var, orient="horizontal",
                          command=moved if step else None)
        scale.grid(row=row, column=1, sticky="ew", pady=2)

        def show(*_):
            try:
                value.configure(text=fmt.format(var.get()))
            except (tk.TclError, ValueError):
                pass

        var.trace_add("write", show)
        show()

    def _build_gene_tab(self, parent, sections) -> list[tuple[ttk.Frame, tuple | None]]:
        """One frame per section, so sections that don't apply to a kind can be hidden."""
        parent.columnconfigure(0, weight=1)
        frames = []
        for i, (title, kinds, controls) in enumerate(sections):
            box = ttk.Frame(parent)
            box.grid(row=i, column=0, sticky="ew", pady=(8, 0))
            box.columnconfigure(0, minsize=130)
            box.columnconfigure(1, weight=1)
            frames.append((box, kinds))
            row = self._section(box, 0, title)
            for label, path, kind, extra in controls:
                if kind == "float":
                    var = tk.DoubleVar()
                    self._slider(box, row, label, var, *extra)
                elif kind == "int":
                    var = tk.IntVar()
                    self._slider(box, row, label, var, *extra, fmt="{:.0f}", step=1)
                elif kind == "choice":
                    var = tk.StringVar()
                    ttk.Label(box, text=label).grid(row=row, column=0, sticky="w", padx=(8, 10), pady=2)
                    ttk.Combobox(box, textvariable=var, values=extra, state="readonly", width=12) \
                        .grid(row=row, column=1, sticky="w", pady=2)
                elif kind == "bool":
                    var = tk.BooleanVar()
                    ttk.Checkbutton(box, text=label, variable=var) \
                        .grid(row=row, column=0, columnspan=2, sticky="w", padx=(8, 0), pady=2)
                else:  # color override
                    var = tk.StringVar()
                    self._color_row(box, row, label, path, var, extra)
                var.trace_add("write", lambda *_: self.gene_changed())
                self.gene_vars[path] = var
                row += 1
            if title == "Body":
                ttk.Button(box, text="Pick body colour...", command=self.pick_body_color) \
                    .grid(row=row, column=1, sticky="w", pady=(4, 0))
                sw = tk.Label(box, width=4, relief="solid", borderwidth=1)
                sw.grid(row=row, column=2, sticky="e", pady=(4, 0))
                self.swatches["body"] = (sw, BASE)
                row += 1
        return frames

    def _show_kind_controls(self, kind: str):
        """Show only the sections that apply to ``kind``; hide tabs left empty."""
        for tab, frames in self.gene_tabs:
            visible = 0
            for box, kinds in frames:
                if kinds is None or kind in kinds:
                    box.grid()
                    visible += 1
                else:
                    box.grid_remove()
            if visible:
                self.notebook.add(tab)  # re-adding a hidden tab shows it in its old place
            else:
                self.notebook.hide(tab)

    def _color_row(self, parent, row, label, path, var, slot):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(8, 10), pady=2)
        cell = ttk.Frame(parent)
        cell.grid(row=row, column=1, columnspan=2, sticky="ew", pady=2)
        sw = tk.Label(cell, width=4, relief="solid", borderwidth=1, cursor="hand2")
        sw.pack(side="left")
        mode = ttk.Label(cell, width=7)
        mode.pack(side="left", padx=6)

        def pick(_=None):
            current = var.get() or self._palette_hex(slot)
            rgb, hx = colorchooser.askcolor(color=current, parent=self.root, title=label)
            if hx:
                var.set(hx)

        sw.bind("<Button-1>", pick)
        ttk.Button(cell, text="Choose...", command=pick).pack(side="left")
        ttk.Button(cell, text="Auto", command=lambda: var.set("")).pack(side="left", padx=(4, 0))
        var.trace_add("write", lambda *_: mode.configure(text="custom" if var.get() else "auto"))
        self.swatches[path] = (sw, slot)

    def _int_row(self, parent, row, label, var, lo, hi, hint=""):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(8, 10), pady=3)
        ttk.Spinbox(parent, textvariable=var, from_=lo, to=hi, width=8).grid(row=row, column=1, sticky="w", pady=3)
        if hint:
            ttk.Label(parent, text=hint, foreground="#666").grid(row=row, column=2, sticky="w")

    def _build_sprite_tab(self, p):
        p.columnconfigure(2, weight=1)
        row = self._section(p, 0, "Canvas")
        self._int_row(p, row, "Width (px)", self.width, 8, 256)
        self._int_row(p, row + 1, "Height (px)", self.height, 8, 256)
        sizes = ttk.Frame(p)
        sizes.grid(row=row + 2, column=0, columnspan=3, sticky="w", padx=8, pady=(2, 0))
        for n in (16, 24, 32, 48, 64, 96):
            ttk.Button(sizes, text=str(n), width=4,
                       command=lambda n=n: (self.width.set(n), self.height.set(n))).pack(side="left", padx=(0, 3))
        ttk.Checkbutton(p, text="Scale canvas with animal size (sharks get bigger sprites)",
                        variable=self.scale_by_size).grid(row=row + 3, column=0, columnspan=3, sticky="w", padx=8, pady=(4, 0))
        row += 1
        row = self._section(p, row + 3, "Animation")
        self._int_row(p, row, "Swim frames", self.swim_frames, 2, 48, "per tail beat")
        self._int_row(p, row + 1, "Turn frames", self.turn_frames, 2, 64, "per turn-around")
        self._int_row(p, row + 2, "Frames / sec", self.fps, 1, 60)
        row = self._section(p, row + 3, "Camera")
        p.columnconfigure(1, weight=1)
        self._slider(p, row, "Tilt", self.pitch, 0.0, 0.8)
        ttk.Label(p, text="Higher tilt shows more of the fish's back.", foreground="#666") \
            .grid(row=row + 1, column=0, columnspan=3, sticky="w", padx=8)

    def _build_export_tab(self, p):
        p.columnconfigure(1, weight=1)
        row = self._section(p, 0, "Where")
        ttk.Label(p, text="File name").grid(row=row, column=0, sticky="w", padx=(8, 10), pady=3)
        ttk.Entry(p, textvariable=self.name).grid(row=row, column=1, columnspan=2, sticky="ew", pady=3)
        ttk.Label(p, text="Folder").grid(row=row + 1, column=0, sticky="w", padx=(8, 10), pady=3)
        ttk.Entry(p, textvariable=self.out_dir).grid(row=row + 1, column=1, sticky="ew", pady=3)
        ttk.Button(p, text="Browse...", command=self.browse_out).grid(row=row + 1, column=2, padx=(6, 0))

        row = self._section(p, row + 2, "What")
        for i, f in enumerate(FORMATS):
            ttk.Checkbutton(p, text=FORMAT_LABELS[f], variable=self.formats[f]) \
                .grid(row=row + i, column=0, columnspan=3, sticky="w", padx=8, pady=1)
        row += len(FORMATS)

        row = self._section(p, row, "Options")
        self._int_row(p, row, "GIF zoom", self.gif_scale, 1, 32, "x")
        ttk.Checkbutton(p, text="Solid GIF background", variable=self.gif_bg_on) \
            .grid(row=row + 1, column=0, columnspan=2, sticky="w", padx=8, pady=3)
        bg_cell = ttk.Frame(p)
        bg_cell.grid(row=row + 1, column=2, sticky="w")
        self.bg_swatch = tk.Label(bg_cell, width=4, relief="solid", borderwidth=1, bg=self.gif_bg.get(), cursor="hand2")
        self.bg_swatch.pack(side="left")
        self.bg_swatch.bind("<Button-1>", lambda e: self.pick_gif_bg())
        self._int_row(p, row + 2, "PNG frame zoom", self.frame_scale, 1, 32, "x")
        self._int_row(p, row + 3, "Sheet padding", self.padding, 0, 16, "px")
        ttk.Label(p, text="Godot folder").grid(row=row + 4, column=0, sticky="w", padx=(8, 10), pady=3)
        ttk.Entry(p, textvariable=self.godot_dir).grid(row=row + 4, column=1, columnspan=2, sticky="ew", pady=3)

        btns = ttk.Frame(p)
        btns.grid(row=row + 5, column=0, columnspan=3, sticky="w", pady=(14, 0), padx=8)
        ttk.Button(btns, text="Export fish", style="Big.TButton", command=self.export).pack(side="left")
        ttk.Button(btns, text="Open folder", command=self.open_out).pack(side="left", padx=8)

    def _build_preview(self, parent):
        bar = ttk.Frame(parent, padding=(8, 8, 8, 4))
        bar.pack(fill="x")
        ttk.Label(bar, text="Animation").pack(side="left")
        self.anim_box = ttk.Combobox(bar, textvariable=self.anim, values=(SHOWCASE, *SHOWCASE_ORDER[:1]),
                                     state="readonly", width=20)
        self.anim_box.pack(side="left", padx=(4, 12))
        ttk.Label(bar, text="Zoom").pack(side="left")
        ttk.Combobox(bar, textvariable=self.zoom, values=("Fit", *[f"{z}x" for z in (1, 2, 3, 4, 6, 8, 12, 16)]),
                     state="readonly", width=5).pack(side="left", padx=(4, 12))
        ttk.Label(bar, text="Background").pack(side="left")
        ttk.Combobox(bar, textvariable=self.background, values=BACKGROUNDS, state="readonly",
                     width=8).pack(side="left", padx=(4, 12))
        ttk.Checkbutton(bar, text="Play", variable=self.playing).pack(side="left")

        self.view = tk.Canvas(parent, highlightthickness=0, background="#1a3a5c")
        self.view.pack(fill="both", expand=True, padx=8)
        self.view.bind("<Configure>", lambda e: self.rebuild_frames())

        foot = ttk.Frame(parent, padding=(8, 4, 8, 8))
        foot.pack(fill="x")
        ttk.Label(foot, textvariable=self.status).pack(side="left")
        ttk.Button(foot, text="Export fish", style="Big.TButton", command=self.export).pack(side="right")

    # ------------------------------------------------------------------ genome <-> controls
    def load_type(self):
        """Build the genome for the chosen species/body type/seed and show it in the controls."""
        kind, species = self._kind(), self.species.get()
        # Body types are a fish thing, and presets fix their own.
        self.arch_box.configure(state="readonly" if kind == "fish" and species == RANDOM else "disabled")
        try:
            seed = int(self.seed.get())
        except (tk.TclError, ValueError):
            return
        arch = self.archetype.get()
        self.base_genome = make_genome(species, seed, None if arch == ANY or species != RANDOM else arch, kind=kind)
        self._show_kind_controls(kind)
        self._loading = True
        try:
            for path, var in self.gene_vars.items():
                value = _get(self.base_genome, path)
                var.set("" if value is None else value)
        finally:
            self._loading = False
        if self._name_is_auto:
            self._set_auto_name()
        self.gene_changed()

    def genome_from_controls(self) -> FishGenome:
        g = copy.deepcopy(self.base_genome)
        g.seed = int(self.seed.get())
        for path, var in self.gene_vars.items():
            value = var.get()
            if path.endswith(("_color",)):
                value = value or None
            _set(g, path, value)
        for fin in (g.dorsal, g.anal, g.dorsal2, g.wings):
            fin.end = max(fin.end, fin.start + 0.05)
        return g

    def gene_changed(self):
        if self._loading:
            return
        self._refresh_swatches()
        self.request_render()

    def _palette_hex(self, slot: int) -> str:
        try:
            return _hex(build_palette(self.genome_from_controls().colors).colors[slot])
        except Exception:
            return "#808080"

    def _refresh_swatches(self):
        try:
            pal = build_palette(self.genome_from_controls().colors).colors
        except Exception:
            return
        for sw, slot in self.swatches.values():
            sw.configure(bg=_hex(pal[slot]))

    def pick_body_color(self):
        rgb, hx = colorchooser.askcolor(color=self._palette_hex(BASE), parent=self.root, title="Body colour")
        if not rgb:
            return
        h, s, v = colorsys.rgb_to_hsv(*(c / 255 for c in rgb))
        self._loading = True
        self.gene_vars["colors.hue"].set(round(h, 3))
        self.gene_vars["colors.saturation"].set(round(s, 3))
        self._loading = False
        self.gene_vars["colors.value"].set(round(max(v, 0.15), 3))

    # ------------------------------------------------------------------ type buttons
    def random_seed(self):
        self.seed.set(random.randrange(1_000_000))

    def surprise(self):
        self._batch = True  # one regeneration, not four
        kind = random.choice(KINDS)
        self.kind.set(kind_label(kind))
        self.species.set(random.choice(species_for(kind)))
        self.archetype.set(ANY)
        self._batch = False
        self.random_seed()

    def _kind(self) -> str:
        return kind_from_label(self.kind.get())

    def _kind_changed(self):
        """New kind: offer its species and start from a random animal of that kind."""
        kind = self._kind()
        self.species_box.configure(values=species_for(kind))
        if self.species.get() not in species_for(kind):
            batch, self._batch = self._batch, True
            self.species.set(RANDOM)
            self._batch = batch
        if not self._batch:
            self.load_type()

    def _type_changed(self):
        if not self._batch:
            self.load_type()

    def _set_auto_name(self):
        self._setting_name = True
        try:
            species = self.species.get()
            stem = self._kind() if species == RANDOM else species.lower().replace(" ", "_").replace("-", "_")
            self.name.set(f"{stem}_{self.seed.get()}")
        except tk.TclError:
            pass
        self._setting_name = False

    def _name_edited(self):
        if not getattr(self, "_setting_name", False):
            self._name_is_auto = not self.name.get().strip()

    # ------------------------------------------------------------------ rendering
    def sprite_opts(self) -> dict:
        try:
            opts = dict(width=int(self.width.get()), height=int(self.height.get()),
                        swim_frames=int(self.swim_frames.get()), turn_frames=int(self.turn_frames.get()),
                        fps=float(self.fps.get()), pitch=float(self.pitch.get()),
                        scale_by_size=bool(self.scale_by_size.get()))
        except (tk.TclError, ValueError):
            raise ValueError("Sprite settings must be numbers")
        if not (8 <= opts["width"] <= 512 and 8 <= opts["height"] <= 512):
            raise ValueError("Size must be between 8 and 512 pixels")
        if opts["swim_frames"] < 2 or opts["turn_frames"] < 2 or opts["fps"] <= 0:
            raise ValueError("Need at least 2 frames per animation and fps above 0")
        return opts

    def request_render(self):
        if self._loading:
            return
        if self._render_after:
            self.root.after_cancel(self._render_after)
        self._render_after = self.root.after(180, self._start_render)

    def _start_render(self):
        self._render_after = None
        if self.base_genome is None:
            return
        try:
            genome = self.genome_from_controls()
            opts = self.sprite_opts()
        except (ValueError, tk.TclError) as e:
            self.status.set(str(e))
            return
        self._job += 1
        with self._lock:
            self._pending = (self._job, genome, opts)
        self._wake.set()
        self.status.set("Rendering...")

    def _worker(self):
        while True:
            self._wake.wait()
            with self._lock:
                job, self._pending = self._pending, None
                self._wake.clear()
            if job is None:
                continue
            job_id, genome, opts = job
            t0 = time.perf_counter()
            try:
                sprite = generate(genome.seed, genome=genome, **opts)
                self._results.put((job_id, sprite, time.perf_counter() - t0, None))
            except Exception as e:  # show render errors in the status bar instead of dying
                self._results.put((job_id, None, 0.0, e))

    def _poll(self):
        try:
            while True:
                job_id, sprite, dt, err = self._results.get_nowait()
                if job_id < self._shown_job:
                    continue
                self._shown_job = job_id
                if err:
                    self.status.set(f"Render failed: {err}")
                else:
                    self.sprite = sprite
                    names = [n for n in SHOWCASE_ORDER if n in sprite.animations]
                    self.anim_box.configure(values=(SHOWCASE, *names))
                    if self.anim.get() not in (SHOWCASE, *names):
                        self.anim.set(SHOWCASE)
                    w, h = sprite.size
                    busy = " (updating...)" if job_id < self._job else ""
                    self.status.set(f"{w}x{h} sprite, {sum(len(a.frames) for a in sprite.animations.values())} "
                                    f"frames, rendered in {dt:.2f}s{busy}")
                    self.rebuild_frames()
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    def _current_frames(self):
        anims = self.sprite.animations
        if self.anim.get() == SHOWCASE:
            return self.sprite.showcase(swim_loops=2), anims["swim_right"].fps
        a = anims[self.anim.get()]
        return a.frames, a.fps

    def rebuild_frames(self):
        if self.sprite is None:
            return
        frames, _ = self._current_frames()
        w, h = self.sprite.size
        cw, ch = max(1, self.view.winfo_width()), max(1, self.view.winfo_height())
        if self.zoom.get() == "Fit":
            zoom = max(1, min((cw - 40) // w, (ch - 40) // h))
        else:
            zoom = int(self.zoom.get().rstrip("x"))
        self._frames = [ImageTk.PhotoImage(Image.fromarray(f).resize((w * zoom, h * zoom), Image.NEAREST))
                        for f in frames]
        self._frame_i %= len(self._frames)
        self._draw_background(cw, ch)
        self.view.delete("fish")
        self.view.create_image(cw // 2, ch // 2, image=self._frames[self._frame_i], tags="fish")

    def _draw_background(self, cw, ch):
        self.view.delete("bg")
        kind = self.background.get()
        if kind == "Water":
            top, bot = (52, 112, 158), (12, 34, 62)
            band = 6
            for y in range(0, ch, band):
                t = y / max(1, ch - 1)
                c = _hex(round(a + (b - a) * t) for a, b in zip(top, bot))
                self.view.create_rectangle(0, y, cw, y + band, fill=c, outline="", tags="bg")
        elif kind == "Checker":
            self.view.configure(background="#e8e8e8")
            n = 16
            for y in range(0, ch, n):
                for x in range((y // n) % 2 * n, cw, n * 2):
                    self.view.create_rectangle(x, y, x + n, y + n, fill="#c8c8c8", outline="", tags="bg")
            return
        self.view.configure(background={"Water": "#1a3a5c", "Dark": "#16161c", "Light": "#f4f4f0"}[kind])

    def _tick(self):
        delay = 100
        if self.sprite is not None and self._frames:
            _, fps = self._current_frames()
            delay = max(10, round(1000 / fps))
            if self.playing.get():
                self._frame_i = (self._frame_i + 1) % len(self._frames)
                self.view.itemconfigure("fish", image=self._frames[self._frame_i])
        self.root.after(delay, self._tick)

    # ------------------------------------------------------------------ export
    def browse_out(self):
        d = filedialog.askdirectory(initialdir=self.out_dir.get(), parent=self.root)
        if d:
            self.out_dir.set(d)

    def pick_gif_bg(self):
        rgb, hx = colorchooser.askcolor(color=self.gif_bg.get(), parent=self.root, title="GIF background")
        if hx:
            self.gif_bg.set(hx)
            self.bg_swatch.configure(bg=hx)
            self.gif_bg_on.set(True)

    def open_out(self):
        d = Path(self.out_dir.get())
        d.mkdir(parents=True, exist_ok=True)
        _open_folder(d)

    def export(self):
        formats = [f for f, v in self.formats.items() if v.get()]
        if not formats:
            messagebox.showwarning("Nothing to export", "Tick at least one format on the Export tab.", parent=self.root)
            return
        name = "".join(c for c in self.name.get().strip() if c.isalnum() or c in "-_ .") or f"fish_{self.seed.get()}"
        try:
            genome = self.genome_from_controls()
            opts = self.sprite_opts()
            gif_scale, frame_scale, padding = int(self.gif_scale.get()), int(self.frame_scale.get()), int(self.padding.get())
        except (ValueError, tk.TclError) as e:
            messagebox.showerror("Can't export", str(e), parent=self.root)
            return
        out = Path(self.out_dir.get()) / name
        self.root.configure(cursor="watch")
        self.root.update_idletasks()
        try:
            sprite = generate(genome.seed, genome=genome, **opts)
            files = export_all(sprite, out, formats, gif_scale=gif_scale, frame_scale=frame_scale,
                               background=self.gif_bg.get() if self.gif_bg_on.get() else None,
                               padding=padding, godot_dir=self.godot_dir.get(), stem=name)
        except Exception as e:
            messagebox.showerror("Export failed", str(e), parent=self.root)
            return
        finally:
            self.root.configure(cursor="")
        self.status.set(f"Exported {len(files)} files to {out}")
        if messagebox.askyesno("Exported", f"Wrote {len(files)} files to\n{out}\n\nOpen the folder?", parent=self.root):
            _open_folder(out)


def run(seed: int | None = None, species: str = RANDOM, out: Path = Path("out"), kind: str = "fish") -> int:
    if sys.platform.startswith("win"):
        try:  # crisp text on high-DPI screens
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    FishStudio(root, seed=seed, species=species, out=out, kind=kind)
    root.mainloop()
    return 0
