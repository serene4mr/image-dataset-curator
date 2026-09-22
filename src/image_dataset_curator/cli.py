import os
import sys
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from image_dataset_curator.pipeline import DatasetFilteringPipeline
from image_dataset_curator.utils.io import export_selected_images


console = Console()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-dataset-curator",
        description="🚀 Multi-Stage Image Dataset Deduplication and Diversity Curation for Computer Vision & Semantic Segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # I/O Arguments
    parser.add_argument('--input-dir', '-i', type=str, required=True, help="Path to input directory containing raw images")
    parser.add_argument('--output-dir', '-o', type=str, required=True, help="Path to output directory")
    parser.add_argument('--extensions', type=str, default="png,jpg,jpeg", help="Comma-separated image extensions")
    parser.add_argument('--action', type=str, choices=['copy', 'symlink', 'list'], default='copy',
                        help="Export action: 'copy', 'symlink' (zero disk overhead), or 'list' (TXT/JSON metadata only)")

    # Mode: Auto vs Budget
    parser.add_argument('--mode', type=str, choices=['auto', 'budget'], default='auto',
                        help="'auto': App determines optimal frame count. 'budget': Uses user-specified --target-pct or --target-count.")

    # Auto Mode Sensitivity
    parser.add_argument('--auto-sensitivity', type=str, choices=['high', 'medium', 'low'], default='medium',
                        help="Auto mode sensitivity: 'high' (~8-12%%), 'medium' (~4-7%%), 'low' (~1-3%%)")

    # Budget Mode Parameters
    parser.add_argument('--target-pct', '-p', type=float, default=None, help="Target percentage in budget mode (e.g., 10 for 10%%)")
    parser.add_argument('--target-count', '-k', type=int, default=None, help="Exact target number of images in budget mode (e.g., 800)")

    # Stage Controls
    parser.add_argument('--disable-stage1', action='store_true', help="Disable Stage 1 (Fixed step subsampling)")
    parser.add_argument('--stage1-step', type=int, default=3, help="Step size for Stage 1 (e.g., 3 takes 1 in every 3 frames)")

    parser.add_argument('--disable-stage2', action='store_true', help="Disable Stage 2 (Sequential similarity filter)")
    parser.add_argument('--stage2-metric', type=str, choices=['ssim', 'dhash'], default='ssim', help="Metric for Stage 2")
    parser.add_argument('--stage2-ssim-thresh', type=float, default=0.85, help="SSIM threshold for Stage 2")
    parser.add_argument('--stage2-dhash-thresh', type=int, default=5, help="dHash Hamming distance threshold")

    parser.add_argument('--disable-stage3', action='store_true', help="Disable Stage 3 (Global diversity filter)")
    parser.add_argument('--stage3-model', type=str, choices=['auto', 'dinov2', 'resnet', 'spatial_hist'], default='auto',
                        help="Feature representation for Stage 3 diversity sampling")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Discover files
    valid_exts = tuple('.' + ext.strip().lower() for ext in args.extensions.split(','))
    if not os.path.exists(args.input_dir):
        console.print(f"[bold red]Error:[/bold red] Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)

    all_files = sorted([
        os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
        if f.lower().endswith(valid_exts)
    ])
    total_raw = len(all_files)
    if total_raw == 0:
        console.print(f"[bold red]Error:[/bold red] No images found in '{args.input_dir}' matching extensions: {args.extensions}")
        sys.exit(1)

    if args.target_pct is not None or args.target_count is not None:
        args.mode = 'budget'

    # Print Banner
    banner_text = (
        f"[bold cyan]Input Folder:[/bold cyan] {args.input_dir}\n"
        f"[bold cyan]Total Raw Images:[/bold cyan] {total_raw:,} frames\n"
        f"[bold cyan]Output Folder:[/bold cyan] {args.output_dir}\n"
        f"[bold cyan]Export Action:[/bold cyan] [bold green]{args.action}[/bold green]\n"
        f"[bold cyan]Filter Mode:[/bold cyan] [bold yellow]{args.mode.upper()}[/bold yellow] "
        f"{f'(Sensitivity: {args.auto_sensitivity})' if args.mode == 'auto' else f'(Target budget)'}"
    )
    console.print(Panel(banner_text, title="🚀 [bold magenta]Image Dataset Curator[/bold magenta]", expand=False))

    pipeline = DatasetFilteringPipeline(
        mode=args.mode,
        auto_sensitivity=args.auto_sensitivity,
        target_pct=args.target_pct,
        target_count=args.target_count,
        enable_stage1=not args.disable_stage1,
        stage1_step=args.stage1_step,
        enable_stage2=not args.disable_stage2,
        stage2_metric=args.stage2_metric,
        stage2_ssim_thresh=args.stage2_ssim_thresh,
        stage2_dhash_thresh=args.stage2_dhash_thresh,
        enable_stage3=not args.disable_stage3,
        stage3_model=args.stage3_model,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task_s2 = progress.add_task("[cyan]Stage 2: Sequential SSIM...", total=total_raw // (args.stage1_step if not args.disable_stage1 else 1), visible=False)
        task_s3 = progress.add_task("[magenta]Stage 3: Feature Extraction...", total=100, visible=False)

        def stage2_cb(curr, total, retained):
            progress.update(task_s2, total=total, completed=curr, visible=True, description=f"[cyan]Stage 2: SSIM ({retained:,} retained)...")

        def stage3_cb(curr, total):
            progress.update(task_s3, total=total, completed=curr, visible=True, description=f"[magenta]Stage 3: Feature Extraction ({curr}/{total})...")

        progress_callbacks = {
            "stage2": stage2_cb,
            "stage3": stage3_cb,
        }

        selected_files, stats = pipeline.run(all_files, progress_callbacks=progress_callbacks)

    # Summary Table
    table = Table(title="📊 [bold green]Pipeline Execution Summary[/bold green]", show_header=True, header_style="bold magenta")
    table.add_column("Pipeline Stage", style="cyan")
    table.add_column("Frames Retained", justify="right", style="bold green")
    table.add_column("Stage Details", style="dim")
    table.add_column("Time", justify="right", style="yellow")

    table.add_row("Input Raw Frames", f"{total_raw:,}", "Original video stream", "-")
    if 'stage1' in stats['stages']:
        s1 = stats['stages']['stage1']
        table.add_row("Stage 1 (Fixed Step)", f"{s1['retained']:,}", f"Step = {s1['step']}", f"{s1['duration_sec']:.2f}s")
    if 'stage2' in stats['stages']:
        s2 = stats['stages']['stage2']
        table.add_row("Stage 2 (Similarity)", f"{s2['retained']:,}", f"Metric: {s2['metric']}", f"{s2['duration_sec']:.2f}s")
    if 'stage3' in stats['stages']:
        s3 = stats['stages']['stage3']
        table.add_row("Stage 3 (Diversity)", f"{s3['retained']:,}", f"Method: {s3['method']}", f"{s3['duration_sec']:.2f}s")

    table.add_row(
        "[bold]FINAL OUTPUT[/bold]",
        f"[bold]{len(selected_files):,}[/bold] ({stats['retention_ratio_pct']:.2f}%)",
        f"[bold green]{stats['reduction_ratio_pct']:.2f}% duplicate removed[/bold green]",
        f"[bold]{stats['total_time_sec']:.2f}s[/bold]"
    )
    console.print(table)

    # Export
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        export_task = progress.add_task(f"[green]Exporting {len(selected_files):,} images ({args.action})...", total=len(selected_files))
        
        def export_cb(curr, total):
            progress.update(export_task, completed=curr)
            
        export_selected_images(
            selected_files,
            args.output_dir,
            action=args.action,
            save_meta=True,
            stats=stats,
            progress_callback=export_cb
        )

    console.print(f"\n✨ [bold green]Success![/bold green] Results exported to [bold underline]{args.output_dir}[/bold underline]")
    console.print(f"📄 Frame list: [italic]{os.path.join(args.output_dir, 'selected_frames.txt')}[/italic]")
    console.print(f"📊 JSON report: [italic]{os.path.join(args.output_dir, 'filter_report.json')}[/italic]\n")


if __name__ == '__main__':
    main()
