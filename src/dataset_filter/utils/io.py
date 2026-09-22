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
    """
    os.makedirs(output_dir, exist_ok=True)
    total = len(selected_paths)
    
    if action in ['copy', 'symlink']:
        img_out_dir = os.path.join(output_dir, 'rgb') if action == 'copy' and not output_dir.endswith('rgb') else output_dir
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
        stats['selected_files'] = [os.path.basename(p) for p in selected_paths]
        with open(report_path, 'w') as f:
            json.dump(stats, f, indent=2)
