#!/usr/bin/env python3
"""Episode tagger — one-shot Tkinter GUI: sync + tag + cut, on raw videos.

Opens the raw head MP4 directly from
``data/raw/<session>/top_cam/<file>.MP4``, computes the audio-refined sync
once at startup (re-using ``sync_videos.load_session``), lets the
operator scrub + tag, and on ``Ctrl+S`` writes ``clips.json`` AND cuts
the per-episode triplets into ``data/output/<session>/episode_clips/``.

Timeline arithmetic is simple now: head is the reference camera, so its
local time equals the unified timeline. Exported ``start``/``end`` are
just ``frame / fps_exact`` in unified seconds; the cutter's per-camera
offset handles the wrists.

The scrub bar is clamped to the audio-overlap window so the operator
cannot tag episodes outside the cuttable range (the safety the old
session-overlap clip used to provide).

Usage::

    python3 scripts/episode_tagger.py                       # latest session
    python3 scripts/episode_tagger.py --session "video 2"
    python3 scripts/episode_tagger.py --out path/to/clips.json

Hotkeys: Space play/pause | Left/Right ±1f | Shift+Left/Right ±10f |
Ctrl+Left/Right ±1s | Home/End jump to overlap start/end | I mark IN |
O mark OUT | Backspace clear marks | Enter add episode |
Delete remove selected | Ctrl+S save (writes JSON + cuts triplets).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

# scripts/ is the script's own dir at runtime, so this import resolves.
import sync_videos as syc

# Repo layout: this file lives at scripts/episode_tagger.py.
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "data" / "raw"
EPISODE_CLIPS_DIR = REPO_ROOT / "data" / "output" / "episode_clips"

DISPLAY_WIDTH = 960
SPEED_CHOICES = (0.25, 0.5, 1.0)


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def resolve_session(raw_root: Path, session_name: Optional[str]) -> Path:
    """Return a session dir under raw_root.

    An explicit ``session_name`` is treated as the literal folder name --
    no pattern matching, so arbitrary capture names (e.g. ``"kitchen_take_3"``)
    work. Otherwise fall back to the latest ``video N`` session that has a
    head camera populated.
    """
    if session_name:
        candidate = raw_root / session_name
        if not candidate.is_dir():
            raise SystemExit(f"Session folder not found: {candidate}")
        return candidate
    sessions = syc.discover_sessions(raw_root)
    if not sessions:
        raise SystemExit(f"No 'video N' sessions under {raw_root} "
                         f"(pass --session NAME to use a non-standard folder)")
    for s in reversed(sessions):
        if syc.has_reference_camera(s):
            return s
    raise SystemExit("No session has a head camera populated yet")


def fmt_time(t: float) -> str:
    if t < 0 or t != t:  # NaN
        return "--:--.---"
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:06.3f}"


# ---------------------------------------------------------------------------
# episode model
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    episode_id: int
    name: str
    in_frame: int
    out_frame: int
    task_label: str = ""
    dominant_hand: str = "right"

    def start_unified(self, fps: float) -> float:
        # head is the reference camera -> head time == unified time.
        return self.in_frame / fps

    def end_unified(self, fps: float) -> float:
        return self.out_frame / fps


@dataclass
class TaggerState:
    session_dir: Path
    cameras: Dict[str, "syc.Camera"]   # full sync state, used at cut-on-save
    video_path: Path                   # raw head MP4 (cameras['head'].path)
    out_path: Path
    output_dir: str
    fps: float                         # float(Fraction); display use
    fps_exact: Fraction                # exact rational; export + cut use
    n_frames: int                      # head total frames
    overlap_lo: float                  # unified-time overlap window start
    overlap_hi: float                  # unified-time overlap window end
    lo_frame: int                      # head-frame index for overlap_lo
    hi_frame: int                      # head-frame index for overlap_hi
    mode: str = "head_wrist"           # 'head_only' or 'head_wrist'
    episodes: List[Episode] = field(default_factory=list)
    current_frame: int = 0
    in_frame: Optional[int] = None
    out_frame: Optional[int] = None


# ---------------------------------------------------------------------------
# Tk app
# ---------------------------------------------------------------------------

class TaggerApp:
    def __init__(self, root: tk.Tk, state: TaggerState):
        self.root = root
        self.state = state
        self.cap = cv2.VideoCapture(str(state.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cv2 cannot open {state.video_path}")
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._playing = False
        self._play_job: Optional[str] = None
        self._scrub_in_progress = False
        self._speed = tk.DoubleVar(value=0.5)
        self._build_ui()
        # Start on the first frame inside the overlap window.
        self._render_frame(self.state.lo_frame)
        if self.state.episodes:
            self._refresh_table()
        self._refresh_status()

    # -- UI construction -------------------------------------------------

    def _build_ui(self) -> None:
        self.root.title(f"Episode tagger — {self.state.video_path.name} "
                         f"[{self.state.mode}]")
        self.root.geometry(f"{DISPLAY_WIDTH + 40}x980")

        # Video canvas
        h = int(DISPLAY_WIDTH * 9 / 16)
        self.canvas = tk.Canvas(self.root, width=DISPLAY_WIDTH, height=h,
                                bg="#111", highlightthickness=0)
        self.canvas.pack(padx=10, pady=(10, 4))

        # Status row
        self.status_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.status_var,
                  font=("TkFixedFont", 10)).pack()

        # Scrub bar — clamped to the overlap window so the operator literally
        # cannot tag outside the cuttable range (replaces the safety the old
        # session_overlap preview clip used to provide).
        scrub_frame = ttk.Frame(self.root)
        scrub_frame.pack(fill="x", padx=10, pady=(2, 2))
        self.scrub = ttk.Scale(scrub_frame,
                                from_=self.state.lo_frame,
                                to=max(self.state.hi_frame, self.state.lo_frame + 1),
                                orient="horizontal",
                                command=self._on_scrub)
        self.scrub.pack(fill="x")
        self.scrub.bind("<ButtonPress-1>", lambda _e: self._pause())
        self.scrub.bind("<ButtonPress-1>", self._scrub_start, add="+")
        self.scrub.bind("<ButtonRelease-1>", self._scrub_end)

        # Transport controls
        bar = ttk.Frame(self.root)
        bar.pack(pady=4)
        ttk.Button(bar, text="⏮", width=3,
                    command=lambda: self._goto(self.state.lo_frame)
                    ).pack(side="left", padx=2)
        ttk.Button(bar, text="⏪ -10f", width=7,
                    command=lambda: self._step(-10)).pack(side="left", padx=2)
        ttk.Button(bar, text="◀ -1f", width=6,
                    command=lambda: self._step(-1)).pack(side="left", padx=2)
        self.play_btn = ttk.Button(bar, text="▶ Play", width=8,
                                    command=self._toggle_play)
        self.play_btn.pack(side="left", padx=4)
        ttk.Button(bar, text="▶ +1f", width=6,
                    command=lambda: self._step(1)).pack(side="left", padx=2)
        ttk.Button(bar, text="⏩ +10f", width=7,
                    command=lambda: self._step(10)).pack(side="left", padx=2)
        ttk.Button(bar, text="⏭", width=3,
                    command=lambda: self._goto(self.state.hi_frame)
                    ).pack(side="left", padx=2)
        ttk.Label(bar, text="speed:").pack(side="left", padx=(10, 2))
        ttk.OptionMenu(bar, self._speed, self._speed.get(),
                        *SPEED_CHOICES).pack(side="left")

        # Mark + form row
        form = ttk.LabelFrame(self.root, text="Mark current episode")
        form.pack(fill="x", padx=10, pady=(8, 4))

        self.marks_var = tk.StringVar(value="in: ---     out: ---")
        ttk.Label(form, textvariable=self.marks_var,
                  font=("TkFixedFont", 10)).grid(
                      row=0, column=0, columnspan=4, sticky="w",
                      padx=6, pady=(4, 2))

        ttk.Label(form, text="task_label").grid(row=1, column=0,
                                                sticky="w", padx=6)
        self.label_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.label_var, width=28).grid(
            row=1, column=1, padx=4, pady=2, sticky="w")
        ttk.Label(form, text="hand").grid(row=1, column=2, sticky="e", padx=6)
        self.hand_var = tk.StringVar(value="right")
        ttk.OptionMenu(form, self.hand_var, "right", "right", "left").grid(
            row=1, column=3, sticky="w")

        ttk.Label(form, text="name (auto)").grid(row=2, column=0,
                                                  sticky="w", padx=6)
        self.name_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.name_var, width=18).grid(
            row=2, column=1, padx=4, pady=2, sticky="w")
        ttk.Button(form, text="Mark IN (I)",
                    command=self._mark_in).grid(row=2, column=2, padx=2)
        ttk.Button(form, text="Mark OUT (O)",
                    command=self._mark_out).grid(row=2, column=3, padx=2)
        ttk.Button(form, text="Add episode (Enter)",
                    command=self._add_episode).grid(row=3, column=1,
                                                     padx=4, pady=4,
                                                     sticky="w")
        ttk.Button(form, text="Clear marks",
                    command=self._clear_marks).grid(row=3, column=2, padx=4)

        # Episode table
        table_frame = ttk.LabelFrame(self.root, text="Episodes")
        table_frame.pack(fill="both", expand=True, padx=10, pady=4)
        cols = ("id", "name", "start", "end", "dur", "label", "hand")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings",
                                  height=8)
        for c, w in zip(cols, (40, 90, 90, 90, 70, 220, 60)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w,
                             anchor="w" if c in ("name", "label") else "center")
        self.tree.pack(fill="both", expand=True, side="left")
        sb = ttk.Scrollbar(table_frame, orient="vertical",
                           command=self.tree.yview)
        sb.pack(side="right", fill="y")
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.bind("<<TreeviewSelect>>", self._on_select_row)

        row_btns = ttk.Frame(self.root)
        row_btns.pack(fill="x", padx=10)
        ttk.Button(row_btns, text="Apply form to selected",
                    command=self._apply_form_to_selected).pack(side="left")
        ttk.Button(row_btns, text="Delete selected (Del)",
                    command=self._delete_selected).pack(side="left", padx=6)
        ttk.Button(row_btns, text="Go to IN of selected",
                    command=lambda: self._goto_selected(field="in")
                    ).pack(side="left", padx=6)
        ttk.Button(row_btns, text="Go to OUT of selected",
                    command=lambda: self._goto_selected(field="out")
                    ).pack(side="left", padx=2)

        # Footer (save)
        foot = ttk.Frame(self.root)
        foot.pack(fill="x", padx=10, pady=(6, 10))
        self.out_var = tk.StringVar(value=f"Out: {self.state.out_path}")
        ttk.Label(foot, textvariable=self.out_var).pack(side="left")
        self.save_btn = ttk.Button(
            foot, text="Save + Cut (Ctrl+S)", command=self._save)
        self.save_btn.pack(side="right")
        ttk.Button(foot, text="Browse…",
                    command=self._pick_out_path).pack(side="right", padx=6)

        # Hotkeys
        self.root.bind("<space>", lambda _e: self._toggle_play())
        self.root.bind("<Left>", lambda _e: self._step(-1))
        self.root.bind("<Right>", lambda _e: self._step(1))
        self.root.bind("<Shift-Left>", lambda _e: self._step(-10))
        self.root.bind("<Shift-Right>", lambda _e: self._step(10))
        self.root.bind("<Control-Left>",
                       lambda _e: self._step(-int(round(self.state.fps))))
        self.root.bind("<Control-Right>",
                       lambda _e: self._step(int(round(self.state.fps))))
        self.root.bind("<Home>", lambda _e: self._goto(self.state.lo_frame))
        self.root.bind("<End>", lambda _e: self._goto(self.state.hi_frame))
        self.root.bind("<KeyPress-i>", lambda _e: self._mark_in())
        self.root.bind("<KeyPress-I>", lambda _e: self._mark_in())
        self.root.bind("<KeyPress-o>", lambda _e: self._mark_out())
        self.root.bind("<KeyPress-O>", lambda _e: self._mark_out())
        self.root.bind("<BackSpace>", lambda _e: self._clear_marks())
        self.root.bind("<Return>", lambda _e: self._add_episode())
        self.root.bind("<Delete>", lambda _e: self._delete_selected())
        self.root.bind("<Control-s>", lambda _e: self._save())

    # -- video display ---------------------------------------------------

    def _render_frame(self, idx: int) -> None:
        # Clamp to the overlap window so step/scrub cannot escape.
        idx = max(self.state.lo_frame,
                  min(idx, self.state.hi_frame))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok:
            return
        # 4K -> 960 wide
        h, w = frame.shape[:2]
        scale = DISPLAY_WIDTH / w
        disp = cv2.resize(frame, (DISPLAY_WIDTH, int(h * scale)),
                          interpolation=cv2.INTER_AREA)
        disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(disp)
        self._photo = ImageTk.PhotoImage(img)
        self.canvas.config(height=img.height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.state.current_frame = idx
        # keep scale in sync without re-firing _on_scrub
        if abs(float(self.scrub.get()) - idx) > 0.5:
            self.scrub.set(idx)
        self._refresh_status()

    def _refresh_status(self) -> None:
        # head is the reference camera -> t_video == t_unified, so only show one.
        f = self.state.current_frame
        fps = self.state.fps
        t = f / fps if fps else 0.0
        self.status_var.set(
            f"t {fmt_time(t)}  |  frame {f} / {self.state.hi_frame}  |  "
            f"overlap [{fmt_time(self.state.overlap_lo)}, "
            f"{fmt_time(self.state.overlap_hi)}]  |  fps={fps:.4f}")
        in_s = "---" if self.state.in_frame is None else (
            f"frame {self.state.in_frame} "
            f"(t {self.state.in_frame / fps:.4f}s)")
        out_s = "---" if self.state.out_frame is None else (
            f"frame {self.state.out_frame} "
            f"(t {self.state.out_frame / fps:.4f}s)")
        self.marks_var.set(f"in: {in_s}     out: {out_s}")

    # -- transport -------------------------------------------------------

    def _step(self, delta: int) -> None:
        self._render_frame(self.state.current_frame + delta)

    def _goto(self, idx: int) -> None:
        self._render_frame(idx)

    def _toggle_play(self) -> None:
        if self._playing:
            self._pause()
        else:
            self._play()

    def _play(self) -> None:
        self._playing = True
        self.play_btn.config(text="⏸ Pause")
        self._tick()

    def _pause(self) -> None:
        self._playing = False
        self.play_btn.config(text="▶ Play")
        if self._play_job is not None:
            self.root.after_cancel(self._play_job)
            self._play_job = None

    def _tick(self) -> None:
        if not self._playing:
            return
        nxt = self.state.current_frame + 1
        if nxt >= self.state.n_frames:
            self._pause()
            return
        self._render_frame(nxt)
        ms = max(int(1000 / (self.state.fps * self._speed.get())), 1)
        self._play_job = self.root.after(ms, self._tick)

    def _on_scrub(self, value: str) -> None:
        if not self._scrub_in_progress:
            # Programmatic moves (from _render_frame) shouldn't loop back.
            return
        self._render_frame(int(float(value)))

    def _scrub_start(self, _event) -> None:
        self._scrub_in_progress = True

    def _scrub_end(self, _event) -> None:
        self._scrub_in_progress = False

    # -- marks / episodes ------------------------------------------------

    def _mark_in(self) -> None:
        self.state.in_frame = self.state.current_frame
        if (self.state.out_frame is not None
                and self.state.out_frame <= self.state.in_frame):
            self.state.out_frame = None
        self._refresh_status()

    def _mark_out(self) -> None:
        if (self.state.in_frame is not None
                and self.state.current_frame <= self.state.in_frame):
            messagebox.showwarning("Bad OUT",
                                    "OUT must be after IN. Step forward "
                                    "or re-mark IN.")
            return
        self.state.out_frame = self.state.current_frame
        self._refresh_status()

    def _clear_marks(self) -> None:
        self.state.in_frame = None
        self.state.out_frame = None
        self._refresh_status()

    def _next_id(self) -> int:
        used = {ep.episode_id for ep in self.state.episodes}
        i = 1
        while i in used:
            i += 1
        return i

    def _add_episode(self) -> None:
        if self.state.in_frame is None or self.state.out_frame is None:
            messagebox.showwarning("Incomplete",
                                    "Mark IN and OUT before adding.")
            return
        eid = self._next_id()
        name = self.name_var.get().strip() or f"ep{eid:03d}"
        if any(ep.name == name for ep in self.state.episodes):
            messagebox.showwarning("Duplicate", f"name '{name}' already used.")
            return
        ep = Episode(
            episode_id=eid, name=name,
            in_frame=self.state.in_frame, out_frame=self.state.out_frame,
            task_label=self.label_var.get().strip(),
            dominant_hand=self.hand_var.get(),
        )
        self.state.episodes.append(ep)
        self._refresh_table()
        self._clear_marks()
        self.name_var.set("")

    def _refresh_table(self) -> None:
        self.tree.delete(*self.tree.get_children())
        fps = self.state.fps
        for ep in self.state.episodes:
            s = ep.start_unified(fps)
            e = ep.end_unified(fps)
            self.tree.insert("", "end", iid=str(ep.episode_id),
                              values=(f"{ep.episode_id:03d}", ep.name,
                                      f"{s:.4f}", f"{e:.4f}",
                                      f"{e - s:.4f}",
                                      ep.task_label, ep.dominant_hand))

    def _on_select_row(self, _event=None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        ep = self._episode_by_iid(sel[0])
        if ep is None:
            return
        self.label_var.set(ep.task_label)
        self.hand_var.set(ep.dominant_hand)
        self.name_var.set(ep.name)

    def _episode_by_iid(self, iid: str) -> Optional[Episode]:
        try:
            eid = int(iid)
        except ValueError:
            return None
        return next((ep for ep in self.state.episodes
                     if ep.episode_id == eid), None)

    def _apply_form_to_selected(self) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        ep = self._episode_by_iid(sel[0])
        if ep is None:
            return
        new_name = self.name_var.get().strip() or ep.name
        if new_name != ep.name and any(
                e.name == new_name for e in self.state.episodes):
            messagebox.showwarning("Duplicate", f"name '{new_name}' in use.")
            return
        ep.name = new_name
        ep.task_label = self.label_var.get().strip()
        ep.dominant_hand = self.hand_var.get()
        self._refresh_table()

    def _delete_selected(self) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        eid = int(sel[0])
        self.state.episodes = [e for e in self.state.episodes
                                if e.episode_id != eid]
        self._refresh_table()

    def _goto_selected(self, field: str) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        ep = self._episode_by_iid(sel[0])
        if ep is None:
            return
        self._goto(ep.in_frame if field == "in" else ep.out_frame)

    # -- save ------------------------------------------------------------

    def _pick_out_path(self) -> None:
        p = filedialog.asksaveasfilename(
            initialdir=str(self.state.out_path.parent),
            initialfile=self.state.out_path.name,
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if p:
            self.state.out_path = Path(p)
            self.out_var.set(f"Out: {self.state.out_path}")

    def _save(self) -> None:
        cfg = build_clips_json(self.state)
        if not cfg["clips"]:
            if not messagebox.askyesno(
                    "Empty",
                    "No episodes to save. Write empty clips.json without cutting?"):
                return
            self.state.out_path.parent.mkdir(parents=True, exist_ok=True)
            self.state.out_path.write_text(json.dumps(cfg, indent=2))
            messagebox.showinfo("Saved",
                                 f"Wrote empty cut sheet to {self.state.out_path}")
            return

        # 1. Write the cut sheet.
        self.state.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.state.out_path.write_text(json.dumps(cfg, indent=2))

        # 2. Cut per-episode triplets via the same frame-exact cutter
        #    sync_videos --config uses. Cuts can take seconds to minutes;
        #    update the status bar and disable Save during the loop.
        out_base = (REPO_ROOT / "data" / "output" / cfg["output_dir"])
        self.save_btn.config(state="disabled")
        try:
            failures = cut_episodes(self.state, cfg, out_base,
                                    progress=self._on_cut_progress)
        finally:
            self.save_btn.config(state="normal")
            self.status_var.set(self.status_var.get().split("  ←")[0])

        msg = (f"Wrote {len(cfg['clips'])} clip(s) to {self.state.out_path}\n\n"
                f"Cut {len(cfg['clips']) - len(failures)} of "
                f"{len(cfg['clips'])} episode triplets into:\n  {out_base}")
        if failures:
            msg += "\n\nFailed:\n  " + "\n  ".join(failures)
            messagebox.showwarning("Saved with errors", msg)
        else:
            messagebox.showinfo("Saved", msg)

    def _on_cut_progress(self, i: int, total: int, name: str) -> None:
        """Status-bar tick during cut-on-save. Forces a Tk redraw."""
        base = self.status_var.get().split("  ←")[0]
        self.status_var.set(f"{base}  ← cutting {name} ({i}/{total})…")
        self.root.update_idletasks()

    def shutdown(self) -> None:
        self._pause()
        try:
            self.cap.release()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# pure functions (CLI + tests)
# ---------------------------------------------------------------------------

def build_clips_json(state: TaggerState) -> dict:
    """Serialise the episode list using the exact rational fps for snapping.

    Head is the reference camera, so ``start``/``end`` are simply
    ``frame / fps_exact`` in unified seconds; no overlap-window offset.
    """
    fps_exact = state.fps_exact
    clips = []
    for ep in state.episodes:
        start = float(Fraction(ep.in_frame) / fps_exact)
        end = float(Fraction(ep.out_frame) / fps_exact)
        clips.append({
            "name": ep.name,
            "start": round(start, 6),
            "end": round(end, 6),
            "episode_id": ep.episode_id,
            "task_label": ep.task_label,
            "dominant_hand": ep.dominant_hand,
        })
    return {"output_dir": state.output_dir, "clips": clips}


def load_episodes_from_json(path: Path, fps_exact: Fraction) -> List[Episode]:
    """Re-hydrate the episode list from an existing clips.json so re-opening
    the tagger picks up where the operator left off. Inverse of
    ``build_clips_json``: ``in_frame = round(start * fps_exact)``."""
    cfg = json.loads(path.read_text())
    episodes: List[Episode] = []
    for idx, c in enumerate(cfg.get("clips", [])):
        in_frame = int(round(float(c["start"]) * float(fps_exact)))
        out_frame = int(round(float(c["end"]) * float(fps_exact)))
        episodes.append(Episode(
            episode_id=int(c.get("episode_id", idx + 1)),
            name=c.get("name", f"ep{idx + 1:03d}"),
            in_frame=in_frame, out_frame=out_frame,
            task_label=c.get("task_label", ""),
            dominant_hand=c.get("dominant_hand", "right"),
        ))
    return episodes


def cut_episodes(state: TaggerState, cfg: dict, out_base: Path,
                  progress=None) -> List[str]:
    """Run the same frame-exact cutter sync_videos --config would.

    Returns the list of episode names that failed (empty on full success).
    ``progress(i, total, name)`` is called before each cut for UI ticks.
    """
    out_base.mkdir(parents=True, exist_ok=True)
    failures: List[str] = []
    total = len(cfg["clips"])
    for i, clip in enumerate(cfg["clips"], 1):
        name = clip["name"]
        if progress is not None:
            progress(i, total, name)
        try:
            syc.create_clip_batch(
                state.cameras,
                float(clip["start"]), float(clip["end"]),
                out_base, name, verbose=False)
        except Exception as exc:  # noqa: BLE001 — isolate per-episode failures
            print(f"ERROR cutting {name}: {exc}", file=sys.stderr)
            failures.append(f"{name}: {exc}")
    return failures


def init_state(session_dir: Path, out: Optional[Path],
                output_dir: str) -> TaggerState:
    """Sync the session, open the raw head clip, and build the tagger state.

    Sync is computed once via ``sync_videos.load_session`` (audio xcorr),
    then persisted to ``data/raw/<session>/sync.json`` so downstream tools
    pick it up too. The raw head MP4 (head is the reference camera) is the
    scrub source — no separate session-overlap preview needed.
    """
    mode = syc.detect_session_mode(session_dir)
    if mode == "head_only":
        print(f"Loading session '{session_dir.name}' (head_only -- "
              f"no sync needed)…", file=sys.stderr)
        cameras = syc.load_session(session_dir, use_audio=False)
    else:
        print(f"Syncing session '{session_dir.name}' (head_wrist)…",
              file=sys.stderr)
        cameras = syc.load_session(session_dir, use_audio=True)
        syc.write_sync_json(session_dir, cameras)
    head = cameras["head"]
    fps_exact = Fraction(head.fps_num, head.fps_den)
    lo, hi = syc.overlap_window(cameras)
    lo_frame = max(0, int(round(lo * float(fps_exact))))
    hi_frame = min(head.n_frames - 1,
                    int(round(hi * float(fps_exact))))

    if out is None:
        out = session_dir / "clips.json"

    episodes: List[Episode] = []
    if out.is_file():
        try:
            episodes = load_episodes_from_json(out, fps_exact)
            print(f"Loaded {len(episodes)} existing episode(s) from "
                  f"{out}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            print(f"WARN  could not parse existing {out}: {exc}",
                  file=sys.stderr)

    return TaggerState(
        session_dir=session_dir,
        cameras=cameras,
        video_path=head.path,
        out_path=out,
        output_dir=output_dir,
        fps=float(fps_exact),
        fps_exact=fps_exact,
        n_frames=head.n_frames,
        overlap_lo=float(lo),
        overlap_hi=float(hi),
        lo_frame=lo_frame,
        hi_frame=hi_frame,
        mode=mode,
        episodes=episodes,
    )


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--session", help="session name e.g. 'video 2' "
                                      "(default: latest under data/raw/)")
    p.add_argument("--out", help="cut-sheet path "
                                  "(default: data/raw/<session>/clips.json)")
    p.add_argument("--output-dir", default=None,
                   help="output_dir field written into clips.json — picks "
                        "the subfolder of data/output/ to cut into. "
                        "Default: '<session>/episode_clips' so per-session "
                        "outputs do not collide.")
    p.add_argument("--raw-root", default=str(RAW_ROOT),
                   help="override the data/raw root")
    args = p.parse_args(argv)

    session_dir = resolve_session(Path(args.raw_root), args.session)
    output_dir = (args.output_dir if args.output_dir is not None
                  else f"{session_dir.name}/episode_clips")
    out = Path(args.out) if args.out else None
    state = init_state(session_dir, out, output_dir)
    print(f"Loaded raw head {state.video_path}  "
          f"({state.n_frames} frames, fps={state.fps:.4f}, mode={state.mode})")
    win_label = "Overlap window" if state.mode == "head_wrist" else "Head window"
    print(f"{win_label} [{state.overlap_lo:.3f}, {state.overlap_hi:.3f}]s "
          f"-> frames [{state.lo_frame}, {state.hi_frame}]")
    print(f"Cut sheet: {state.out_path}")

    root = tk.Tk()
    app = TaggerApp(root, state)
    root.protocol("WM_DELETE_WINDOW",
                   lambda: (app.shutdown(), root.destroy()))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
