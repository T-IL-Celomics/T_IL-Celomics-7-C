from pathlib import Path
import shutil
import re
import sys
import json
import tifffile as tiff


def natural_t_index(p: Path) -> int:
    m = re.search(r"_(\d+)\.tiff?$", p.name, re.IGNORECASE)
    return int(m.group(1)) if m else 10**18


def stack_timepoints_to_tif(time_files: list[Path], out_tif: Path):
    # stream-write multi-page tif (one page per timepoint) to avoid RAM usage
    first = tiff.imread(time_files[0])
    T = len(time_files)
    shape = [T, *list(first.shape)]
    desc = json.dumps({"shape": shape})

    out_tif.parent.mkdir(parents=True, exist_ok=True)

    with tiff.TiffWriter(out_tif, bigtiff=True) as tw:
        for i, fp in enumerate(time_files):
            frame = tiff.imread(fp)
            if frame.shape != first.shape:
                raise RuntimeError(
                    f"Shape mismatch in {fp}\nExpected {first.shape}, got {frame.shape}"
                )
            tw.write(
                frame,
                contiguous=True,
                description=desc if i == 0 else None
            )


def write_metadata_csv(meta_path: Path, sizeT: int, basename: str, channel_name: str, sizeZ: int):
    # Matches the style you showed: Description,values + SizeT + basename + channel_0_name + SizeZ
    txt = "\n".join(
        [
            "Description,values",
            f"SizeT,{sizeT}",
            f"basename,{basename}",
            f"channel_0_name,{channel_name}",
            f"SizeZ,{sizeZ}",
            "",
        ]
    )
    meta_path.write_text(txt, encoding="utf-8")


def main():
    """
    Input (from your renaming step):
      ACDC_IN_Phase3/field1/B3_1/NIR/Images/field1_B3_NIR_0.tif ... _47.tif

    Output (single experiment folder):
      segmentation_input/Position_1/Images/field1_B3_NIR.tif + field1_B3_metadata.csv
      segmentation_input/Position_2/Images/field1_B3_GREEN.tif + field1_B3_metadata.csv
      ...
    """

    if len(sys.argv) < 2:
        print("Usage: python build_segmentation_input.py <PhaseName> [out_folder_name]")
        print("Example: python build_segmentation_input.py Phase3 segmentation_input")
        sys.exit(1)

    phase_name = sys.argv[1]  # e.g. Phase3
    out_folder_name = sys.argv[2] if len(sys.argv) >= 3 else "segmentation_input"

    script_dir = Path(__file__).parent.parent  # scripts/ -> project root
    renamed_root = script_dir / f"ACDC_IN_{phase_name}"
    out_root = script_dir / out_folder_name

    if not renamed_root.exists():
        raise SystemExit(f"Renamed folder not found: {renamed_root}")

    out_root.mkdir(parents=True, exist_ok=True)

    # We number positions over: (field_folder, field_name_dir, channel)
    # Deterministic order via sorted traversal.
    pos_counter = 0

    # Optional mapping log (helps debugging / later crosswalk)
    map_csv = out_root / "position_map.csv"
    map_lines = ["Position,field_folder,field_name,location,channel,source_images_dir\n"]

    # Traverse: ACDC_IN_PhaseX/<field_folder>/<field_name>/<channel>/Images/*.tif
    for field_folder in sorted([p for p in renamed_root.iterdir() if p.is_dir()]):  # field1, field2, fieldX...
        for field_name_dir in sorted([p for p in field_folder.iterdir() if p.is_dir()]):  # B3_1, C4_2...
            field_name = field_name_dir.name
            # location name (B3 from B3_1)
            location = field_name.rsplit("_", 1)[0] if "_" in field_name else field_name

            for channel_dir in sorted([p for p in field_name_dir.iterdir() if p.is_dir()]):  # NIR, GREEN, ...
                channel_name = channel_dir.name
                images_dir = channel_dir / "Images"
                if not images_dir.exists():
                    continue

                # Collect timepoint tifs created by your renaming script:
                # pattern: <field_folder>_<location>_<channel>_<t>.tif
                rx = re.compile(
                    rf"^{re.escape(field_folder.name)}_{re.escape(location)}_{re.escape(channel_name)}_(\d+)\.tiff?$",
                    re.IGNORECASE,
                )
                time_files = [fp for fp in images_dir.glob("*.tif*") if rx.match(fp.name)]
                if not time_files:
                    continue

                time_files.sort(key=natural_t_index)

                # Build output Position folder
                pos_counter += 1
                pos_folder = out_root / f"Position_{pos_counter}" / "Images"
                pos_folder.mkdir(parents=True, exist_ok=True)

                # Position prefix used inside file names
                # Example: field1_B3
                pos_prefix = f"{field_folder.name}_{location}"
                basename = pos_prefix + "_"  # IMPORTANT: matches Cell-ACDC example (basename ends with "_")

                # Output files inside Position_x/Images
                out_tif = pos_folder / f"{pos_prefix}_{channel_name}.tif"
                meta_csv_path = pos_folder / f"{basename}metadata.csv"  # -> field1_B3_metadata.csv

                print("========================================")
                print(f'Processing field "{field_folder.name}/{field_name}"...')
                print(f'  Processing channel "{channel_name}"...')
                print(f'  Source: {images_dir}')
                print(f'  Target: {pos_folder}')
                print("Saving image files...")

                # Write stacked tif
                stack_timepoints_to_tif(time_files, out_tif)

                # Determine SizeZ for metadata
                sample = tiff.imread(time_files[0])
                sizeZ = 1 if sample.ndim == 2 else sample.shape[0]

                # Write metadata
                write_metadata_csv(
                    meta_path=meta_csv_path,
                    sizeT=len(time_files),
                    basename=basename,
                    channel_name=channel_name,
                    sizeZ=sizeZ,
                )

                # ── Copy TrackPy CSV if present ───────────────────────────
                # Convention: <field_folder>_<location>_trackpy_tracks.csv
                # e.g. field1_B2_trackpy_tracks.csv  lives in project root
                tp_src = script_dir / f"{field_folder.name}_{location}_trackpy_tracks.csv"
                if tp_src.exists():
                    tp_dst = pos_folder / tp_src.name
                    shutil.copy2(tp_src, tp_dst)
                    print(f"  Copied TrackPy CSV: {tp_src.name}")
                else:
                    print(f"  [info] No TrackPy CSV found for {field_folder.name}_{location} (expected: {tp_src.name})")

                map_lines.append(
                    f"{pos_counter},{field_folder.name},{field_name},{location},{channel_name},{images_dir}\n"
                )

    map_csv.write_text("".join(map_lines), encoding="utf-8")

    print("\nDone.")
    print("Output root:", out_root)
    print("Positions created:", pos_counter)
    print("Mapping file:", map_csv)


if __name__ == "__main__":
    main()