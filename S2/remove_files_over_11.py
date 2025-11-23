import os
import re
import argparse

PAT = re.compile(r'^(.+?)_(\d+)\.jpg$', re.IGNORECASE)

def find_and_maybe_delete(root_dir: str, do_delete: bool):
    deleted = 0
    candidates = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            m = PAT.match(fname)
            if not m:
                continue
            num = int(m.group(2))
            if num > 10:
                full = os.path.join(dirpath, fname)
                candidates.append(full)

    if not candidates:
        print("No matching files with number > 11 found.")
        return

    print(f"Found {len(candidates)} files with number > 11:")
    for p in candidates:
        print("  " + p)

    if do_delete:
        for p in candidates:
            try:
                os.remove(p)
                deleted += 1
            except Exception as e:
                print(f"Failed to delete {p}: {e}")
        print(f"Deleted {deleted}/{len(candidates)} files.")
    else:
        print("Dry-run (no files deleted). Use --do-delete to remove files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove .bmp files named like className_XXX.bmp where XXX > 11")
    parser.add_argument("path", nargs="?", default="./data", help="root data folder to scan")
    parser.add_argument("--do-delete", action="store_true", help="actually delete matching files")
    args = parser.parse_args()
    find_and_maybe_delete(args.path, args.do_delete)