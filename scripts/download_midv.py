"""Optional FTP downloader. Host/path defaults are from memory and UNVERIFIED -
check the official MIDV-500 / MIDV-2020 pages (and their license) first.
Manual download into data/raw/ works just as well.

python scripts/download_midv.py --host smartengines.com --path /midv-500/dataset --max_files 6
"""
import argparse
import os
import shutil
from ftplib import FTP
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="smartengines.com")
    ap.add_argument("--path", default="/midv-500/dataset")
    ap.add_argument("--pattern", default="", help="substring filter on file names")
    ap.add_argument("--max_files", type=int, default=6)
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--keep_archives", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ftp = FTP(args.host, timeout=120)
    ftp.login()
    ftp.cwd(args.path)
    names = [n for n in ftp.nlst()
             if args.pattern in n and n.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz"))]
    names = sorted(names)[:args.max_files]
    print("Will download:", names)
    for n in names:
        dst = out / n
        if not dst.exists():
            print("downloading", n)
            with open(dst, "wb") as f:
                ftp.retrbinary(f"RETR {n}", f.write)
        print("extracting", n)
        shutil.unpack_archive(str(dst), str(out))
        if not args.keep_archives:
            os.remove(dst)
    ftp.quit()
    print("done ->", out)


if __name__ == "__main__":
    main()