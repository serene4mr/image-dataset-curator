import os
import shutil
import json
from typing import List, Optional, Callable


def export_selected_images(
    selected_paths: List[str],
    output_dir: str,
    action: str = 'copy',
    save_meta: bool = True,
    stats: Optional[dict] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
):
    """
    Exports filtered images via copy, symlink, or file listing.
    Automatically detects and synchronizes paired depth frames (e.g. from RealSense sessions).
    """
    os.makedirs(output_dir, exist_ok=True)
    total = len(selected_paths)
    paired_depth_count = 0
    
    if action in ['copy', 'symlink']:
        # If output_dir doesn't end with 'rgb', create an 'rgb' subfolder
        img_out_dir = os.path.join(output_dir, 'rgb') if not output_dir.endswith('rgb') else output_dir
        os.makedirs(img_out_dir, exist_ok=True)
        
        for idx, src_path in enumerate(selected_paths):
            fname = os.path.basename(src_path)
            dst_path = os.path.join(img_out_dir, fname)
            
            if os.path.exists(dst_path) or os.path.islink(dst_path):
                os.remove(dst_path)
                
            if action == 'copy':
                shutil.copy2(src_path, dst_path)
            elif action == 'symlink':
                os.symlink(os.path.abspath(src_path), dst_path)

            # Detect and export paired depth file if present (RealSense session structure)
            parent_dir = os.path.dirname(src_path)
            if os.path.basename(parent_dir) == 'rgb':
                session_root = os.path.dirname(parent_dir)
                depth_dir_src = os.path.join(session_root, 'depth')
                if os.path.isdir(depth_dir_src):
                    base_stem = fname.replace('_rgb', '').split('.')[0]
                    candidates = [
                        f"{base_stem}_depth.npy",
                        f"{base_stem}_depth.png",
                        f"{fname.rsplit('.', 1)[0]}.npy",
                        f"{fname.rsplit('.', 1)[0]}.png",
                    ]
                    for d_cand in candidates:
                        depth_src_file = os.path.join(depth_dir_src, d_cand)
                        if os.path.exists(depth_src_file):
                            depth_out_dir = os.path.join(output_dir, 'depth')
                            os.makedirs(depth_out_dir, exist_ok=True)
                            depth_dst_file = os.path.join(depth_out_dir, d_cand)
                            if os.path.exists(depth_dst_file) or os.path.islink(depth_dst_file):
                                os.remove(depth_dst_file)
                            if action == 'copy':
                                shutil.copy2(depth_src_file, depth_dst_file)
                            elif action == 'symlink':
                                os.symlink(os.path.abspath(depth_src_file), depth_dst_file)
                            paired_depth_count += 1
                            break
                
            if progress_callback:
                progress_callback(idx + 1, total)

    # Always write list of selected frames
    list_txt_path = os.path.join(output_dir, 'selected_frames.txt')
    with open(list_txt_path, 'w') as f:
        for p in selected_paths:
            f.write(os.path.basename(p) + '\n')
            
    # Save full filter report
    if save_meta and stats is not None:
        report_path = os.path.join(output_dir, 'filter_report.json')
        stats['selected_count'] = total
        stats['paired_depth_synced'] = paired_depth_count
        stats['selected_files'] = [os.path.basename(p) for p in selected_paths]
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2)
