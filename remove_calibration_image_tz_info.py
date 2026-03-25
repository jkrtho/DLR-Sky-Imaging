import os
import re

def remove_tz_info_from_filename(filename):
    """
    Removes timezone info patterns like "+hhmm", "-hhmm", or "Z" from the end of the stem (before extension).
    E.g. "20250701_051430+0200.webp" -> "20250701_051430.webp"
    """
    stem, ext = os.path.splitext(filename)
    # Remove +hhmm/-hhmm or Z at the end of the stem
    # Handles e.g. ..._051430+0200.webp or ..._051430-0430.webp or ..._051430Z.webp
    new_stem = re.sub(r'([+-][0-2][0-9][0-5][0-9]|Z)$', '', stem)
    return new_stem + ext

def remove_tz_info_in_calibration_images(root_dir="Calibration_images"):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.startswith('.'):
                continue
            new_fname = remove_tz_info_from_filename(fname)
            if new_fname != fname:
                src = os.path.join(dirpath, fname)
                dst = os.path.join(dirpath, new_fname)
                if not os.path.exists(dst):
                    os.rename(src, dst)
                else:
                    print(f"Skipped renaming {src} to {dst} (destination exists)")

if __name__ == '__main__':
    remove_tz_info_in_calibration_images()