# 3D-printing files (STL / FreeCAD)

The MSE-6 chassis CAD and print files are **not committed** to this repository — binary CAD blobs
bloat clone size and, once committed, live in git history forever. They are distributed as a
**GitHub Release asset** instead, and are gitignored here (see `.gitignore`: `*.stl`, `*.FCStd`,
`docs/3D_printing_files/*`).

## Download

Grab the CAD bundle from the **`hardware-v6`** GitHub Release (Releases tab of this repo):

| File | What it is |
| ---- | ---------- |
| `mse6_v6_complete.stl` | Full assembly, print-ready |
| `mse6_v6_top.stl` / `mse6_v6_bottom.stl` | Split halves |
| `Mouse_Droid_Complete.FCStd` | FreeCAD source (editable) |
| `Mouse_Droid_Top.FCStd` / `Mouse_Droid_Bottom.FCStd` | FreeCAD source halves |

> These files previously lived in git history and were purged to shrink the clone — see
> [`docs/runbooks/history-purge.md`](../runbooks/history-purge.md).
