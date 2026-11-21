"""Export per-spectrum VI results to CSV for external audit."""
import csv
import json
import os

WORK = "/data2/wbc/slo/l1_test/eva_work"
OUT = os.path.join(WORK, "vi_results_table.csv")


def main():
    with open(os.path.join(WORK, "vi_results.json")) as fh:
        data = json.load(fh)
    results = data["results"]
    gt = data["ground_truth"]
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "target_type", "index", "targetid",
            "gt_spectype", "gt_z", "gt_quality",
            "ag_spectype", "ag_z", "ag_quality",
            "z_correct", "type_correct", "vi_correct",
            "dz_frac", "flags", "quality_capped", "fallback",
            "evidence", "reasoning",
        ])
        for path, res in sorted(results.items(), key=lambda kv: kv[0]):
            g = gt[path]
            ttype = g["target_type"]
            idx = g["index"]
            z_ok = abs(res["vi_z"] - g["vi_z"]) / (1 + g["vi_z"]) < 0.003
            t_ok = res["vi_spectype"].upper() == g["vi_spectype"].upper()
            w.writerow([
                ttype, idx, res.get("targetid", ""),
                g["vi_spectype"], g["vi_z"], g["vi_quality"],
                res["vi_spectype"], res["vi_z"], res["vi_quality"],
                z_ok, t_ok, z_ok and t_ok,
                round(abs(res["vi_z"] - g["vi_z"]) / (1 + g["vi_z"]), 5),
                ";".join(res.get("flags", [])),
                res.get("quality_capped", False), res.get("fallback", False),
                " | ".join(res.get("evidence", [])),
                res.get("reasoning", ""),
            ])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
