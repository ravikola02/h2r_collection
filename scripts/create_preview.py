#!/usr/bin/env python3
"""
Create a side-by-side preview montage of synchronized clips.

This script combines head + wrist_left + wrist_right videos into a single
visual preview to verify sync alignment and show all three camera views.

Reads per-episode clips from data/output/<session>/episode_clips/ and writes
montages to data/output/<session>/preview/.
"""

import argparse
import subprocess
from pathlib import Path
from typing import Tuple


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1",
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    for line in result.stdout.split("\n"):
        if line.startswith("duration="):
            return float(line.split("=")[1])
    raise ValueError(f"Could not extract duration from {video_path}")


def create_montage(
    head_video: str,
    wrist_left_video: str,
    wrist_right_video: str,
    output_path: str,
    layout: str = "v",  # "v" for vertical stack, "h" for horizontal
    apply_rotation: bool = True,
) -> str:
    """
    Create a montage of three videos side-by-side.
    
    Layout options:
      "v": vertical stack (head on top, wrists below)
           [   HEAD (3840x2160)   ]
           [WRIST_L]  [WRIST_R]
           
      "h": horizontal (head on left, wrists stacked on right)
           [HEAD]  [WRIST_L]
           [HEAD]  [WRIST_R]
    """
    
    # Rotation filter (apply to all cameras to correct -180°)
    rotate_filter = "rotate=180*PI/180" if apply_rotation else "format=yuv420p"
    
    if layout == "v":
        # Vertical stack layout
        # Head: scale to 1920x1080 (downscale from 3840x2160)
        # Wrists: keep 1920x1080
        # Total output: 1920x1440 (1080 + 360 for each wrist)
        # Actually better: 1920x2160 total (head at 1920x1080, wrists below at 1920x1080 combined)
        
        # Better approach: head (1920 wide) + wrists side-by-side below
        # Final: 1920 width, 1080 (head) + 540 (wrists scaled) = 1620 height
        
        filter_complex = (
            f"[0:v] {rotate_filter}, scale=1920:1080 [head]; "
            f"[1:v] {rotate_filter}, scale=960:540 [wleft]; "
            f"[2:v] {rotate_filter}, scale=960:540 [wright]; "
            f"[head][wleft][wright] concat=n=2:v=1:a=0 [v1]; "
            f"[v1][wleft][wright] hstack=inputs=3:height=540 [out]"
        )
        # Actually, let's use a simpler approach: head full width, wrists below side by side
        filter_complex = (
            f"[0:v] {rotate_filter}, scale=1920:1080 [head]; "
            f"[1:v] {rotate_filter}, scale=960:540 [wleft]; "
            f"[2:v] {rotate_filter}, scale=960:540 [wright]; "
            f"[wleft][wright] hstack=inputs=2 [wrists]; "
            f"[head][wrists] vstack=inputs=2 [out]"
        )
    else:  # horizontal layout
        # Head on left, wrists stacked on right
        # Head: 1920x1080 (downscaled from 3840x2160)
        # Wrists: 960x1080 each (scaled to half width)
        # Total: 2880 x 1080
        filter_complex = (
            f"[0:v] {rotate_filter}, scale=1920:1080 [head]; "
            f"[1:v] {rotate_filter}, scale=960:1080 [wleft]; "
            f"[2:v] {rotate_filter}, scale=960:1080 [wright]; "
            f"[wleft][wright] vstack=inputs=2 [wrists]; "
            f"[head][wrists] hstack=inputs=2 [out]"
        )
    
    # Build ffmpeg command (using faster encoder and lower quality for speed)
    cmd = [
        "ffmpeg",
        "-i", head_video,
        "-i", wrist_left_video,
        "-i", wrist_right_video,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-pix_fmt", "yuv420p",
        "-r", "30",
        "-y",
        output_path,
    ]
    
    print(f"Creating {layout}-stack montage: {output_path}")
    print(f"  Filter: {filter_complex}")
    print()
    
    subprocess.run(cmd, check=True, capture_output=True)
    
    return output_path


def main():
    ap = argparse.ArgumentParser(description="Side-by-side preview montage")
    ap.add_argument("--session", help="session name under data/output/ "
                    "(default: first session that has cut clips)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_root = root / "data" / "output"
    if args.session:
        src_dir = out_root / args.session / "episode_clips"
    else:
        candidates = sorted(out_root.glob("*/episode_clips"))
        src_dir = next((d for d in candidates if any(d.glob("*_head.mp4"))),
                       out_root / "_none")
    preview_dir = src_dir.parent / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    # Find all clip sets
    clip_sets = {}
    for f in src_dir.glob("*_head.mp4"):
        clip_name = f.name.replace("_head.mp4", "")
        clip_sets[clip_name] = {
            "head": src_dir / f"{clip_name}_head.mp4",
            "wrist_left": src_dir / f"{clip_name}_wrist_left.mp4",
            "wrist_right": src_dir / f"{clip_name}_wrist_right.mp4",
        }

    if not clip_sets:
        print(f"No clips found in {src_dir.name}/ directory")
        return
    
    print("\n" + "="*80)
    print("CREATING VIDEO MONTAGE PREVIEWS")
    print("="*80 + "\n")
    
    print(f"Found {len(clip_sets)} clip sets:\n")
    for name in sorted(clip_sets.keys()):
        print(f"  {name}")
    
    print("\nGenerating previews (this may take a minute per clip)...\n")
    
    created_previews = {}
    
    for clip_name, videos in sorted(clip_sets.items()):
        # Check if all three videos exist
        if not all(v.exists() for v in videos.values()):
            print(f"⚠ Skipping {clip_name}: missing video files")
            continue
        
        output_vstack = preview_dir / f"{clip_name}_vstack.mp4"
        output_hstack = preview_dir / f"{clip_name}_hstack.mp4"
        
        try:
            # Create vertical stack (head on top, wrists below)
            create_montage(
                str(videos["head"]),
                str(videos["wrist_left"]),
                str(videos["wrist_right"]),
                str(output_vstack),
                layout="v",
                apply_rotation=True,
            )
            created_previews[clip_name] = {
                "vstack": output_vstack,
            }
            
            print(f"✓ Created: {output_vstack.name}\n")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Error creating preview for {clip_name}: {e}\n")
    
    # Summary
    print("\n" + "="*80)
    print("PREVIEW SUMMARY")
    print("="*80 + "\n")
    
    if created_previews:
        print(f"Created {len(created_previews)} preview videos:\n")
        for clip_name, previews in sorted(created_previews.items()):
            vstack_path = previews["vstack"]
            if vstack_path.exists():
                size_mb = vstack_path.stat().st_size / (1024**2)
                print(f"  {vstack_path.name:45} ({size_mb:6.1f} MB)")
        
        print("\n" + "="*80)
        print("PLAYBACK COMMANDS")
        print("="*80 + "\n")
        
        print("View the vertical-stack preview (head top, wrists bottom):\n")
        for clip_name in sorted(created_previews.keys()):
            vstack_path = created_previews[clip_name]["vstack"]
            print(f"  ffplay '{vstack_path.name}'")
        
        print("\n" + "="*80 + "\n")
    else:
        print("No previews could be created")


if __name__ == "__main__":
    main()
