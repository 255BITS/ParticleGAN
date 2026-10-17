"""Extract immutable proposal sources for the documented observation runners.

Only the exact review heads/bases are used. No branch is checked out or changed.
The output cache must stay outside Git.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess

from .capture import write
from .render import read


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inventory",type=Path,default=Path("reports/toy_audit/pull_requests.json"))
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--pr",type=int,nargs="+")
    args=ap.parse_args();receipts=[]
    for row in read(args.inventory):
        if args.pr and row["number"] not in args.pr:continue
        if not args.pr and row["scope"] not in ["New test/counterexample proposal","Existing-data counterexample"]:continue
        for sha in [row["head_sha"],row["base_sha"]]:
            found=subprocess.run(["git","cat-file","-e",sha+"^{commit}"],capture_output=True)
            if found.returncode:subprocess.run(["git","fetch","origin",sha],check=True)
        paths=subprocess.check_output(["git","diff","--name-only","--diff-filter=AMR",row["base_sha"]+"..."+row["head_sha"]],text=True).splitlines()
        for name in paths:
            if Path(name).suffix not in [".py",".md",".json",".yaml",".yml",".toml"]:continue
            content=subprocess.check_output(["git","show",row["head_sha"]+":"+name])
            destination=args.output/str(row["number"])/name
            destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(content)
            receipts.append(dict(pr=row["number"],head_sha=row["head_sha"],path=name,sha256=hashlib.sha256(content).hexdigest()))
        print(f"PR{row['number']} sources extracted at {row['head_sha']}",flush=True)
    write(args.output/"receipts.json",receipts)


if __name__=="__main__":main()
