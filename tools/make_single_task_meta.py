import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="evaluation_examples/test_all.json")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--out", default="evaluation_examples/test_gui_agent_one.json")
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)

    data = json.loads(src.read_text(encoding="utf-8"))

    if args.domain not in data:
        print("[ERROR] Unknown domain:", args.domain)
        print("[INFO] Available domains:")
        for k in data.keys():
            print("  -", k, f"({len(data[k])} tasks)")
        raise SystemExit(1)

    tasks = data[args.domain]

    if args.index < 0 or args.index >= len(tasks):
        print(f"[ERROR] Index out of range: {args.index}")
        print(f"[INFO] Domain '{args.domain}' has {len(tasks)} tasks.")
        raise SystemExit(1)

    task_id = tasks[args.index]
    single = {args.domain: [task_id]}

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(single, indent=2), encoding="utf-8")

    print("[OK] Wrote single-task meta")
    print("domain:", args.domain)
    print("index:", args.index)
    print("task_id:", task_id)
    print("out:", out)


if __name__ == "__main__":
    main()