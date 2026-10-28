# 03 Data format

Step 01 reads two things per system: the **initial conditions** (density grids) and the **labels**
(N_ch and [pT] of the same events from the full hybrid simulation).

## 1. Labels: `y_file`

A text file with one line per event and numbers separated by spaces. Lines starting with `#` are ignored.

| column (default) | content | config key |
|---|---|---|
| 0 | event id (an integer) | `data.y_columns.id` |
| 1 | N_ch: charged-particle multiplicity, pT ≥ 0.15 GeV, \|η\| < 0.5 | `data.y_columns.nch` |
| 2 | [pT]: mean transverse momentum of the same particles, in **GeV** | `data.y_columns.pt` |
| 3 … 10 | further moments from the hydrodynamic output | not used yet (see below) |

This is the format of `Y_eta0p5_pt.dat` of the v6 samples. If your file has another column order,
change `data.y_columns` in your configuration instead of editing the file.

The v6 files have 11 columns. Columns 3–10 hold further moments obtained from the hydrodynamic
evolution. The networks do not use them yet: regressing them is planned for future work. To keep them
in the event table for analysis, declare them in `data.extra_y_columns` (e.g. `{m3: 3, m4: 4}`).
To make one a new target, see [07_adding_inputs.md](07_adding_inputs.md), Level 3. Ask your supervisor
for the exact definition of each column first.

N_ch is an average over many particlizations of the same hydrodynamic event, so it need not be an
integer, and it can be below 1 in the most peripheral events.

## 2. Initial conditions: `x_path`

The 280 × 280 grid (`--grid-max 14 --grid-step 0.1` in TRENTo, 0.1 fm cells) of either the **energy
density** from TRENTo (`data.x_kind: energy`, preferred) or the **entropy density** already converted
from it (`data.x_kind: entropy`). Two layouts are accepted.

**Layout A, one file, one event per line.** Line i holds the 78 400 numbers of event i, row after row.
Line i must belong to label row i: the files are matched **by line number**, so a missing line shifts
every later event. This is the format of the v6 `x_entropy.dat` / `x_energy.dat`.

**Layout B, one file per event (recommended).** A directory with files such as `flattened_IC_17.txt`.
The first number in the file name is the event id, which must equal the id in column 0 of `y_file`
(change `data.x_pattern` if your names differ). Each file holds the 78 400 numbers, as one column or
as a 280 × 280 table. Files are matched **by id**: a missing or extra file only affects that event, and
step 01 lists it.

**Prefer energy density.** With `x_kind: energy`, step 01 computes the entropy density with exactly
the equation of state and interpolation used for v6 (cubic, s = 0 below the table). It also keeps both
grids, so the same prepared data serves entropy and energy runs. If you receive entropy files, ask how
they were made. A linear interpolation, or extrapolation below the table, changes dilute regions by up
to 10 % per pixel.

**The TRENTo parameters must be those of the hybrid events and of the centrality table.** For v6 they
are, apart from the nucleus and the normalization:

```
-p 0.063 -k 1.05 -x 6.28 -w 1.12 -d 1.437425 --grid-max 14.0 --grid-step 0.1     -n 14.2 (Pb Pb) / -n 5.73 (Au Au)
```

If the new samples were made with other parameters, the minimum-bias centrality tables in
`reference/centrality/` no longer apply. Make new ones with `tools/make_centrality_table.py` from a large
TRENTo sample (step 10 B produces one) and set `evaluation.centrality_table`.

## 3. What step 01 checks

| check | what happens |
|---|---|
| every grid has 78 400 numbers | stops with the file and line |
| layout A: as many lines as labels | stops |
| layout B: ids unique; files without labels and labels without files | listed in `data_report.md` |
| NaN/inf or negative values in a grid | event excluded |
| N_ch ≤ 0, [pT] ≤ 0, NaN labels | event excluded (the log target is undefined) |
| total S = 0 (empty grid) | event excluded |
| identical (N_ch, [pT]) in several events | warning: duplicated events? |
| [pT] outside 0.2–2 GeV | warning: wrong column or unit? |
| events also present in the v6 sample | recognized by (N_ch, [pT]); stored as `v6_part` = train / test |

Excluded events and the reason are in `<run_dir>/data/excluded_events.csv`.

## 4. What step 01 writes

```
<run_dir>/data/event_table.csv     idx, event_id, nch, pt, S_entropy, [S_energy], centrality, [v6_idx, v6_part], [extra columns]
<run_dir>/data/grid_entropy.npy    float32 array (N, 280, 280)
<run_dir>/data/grid_energy.npy     float32 array (N, 280, 280)   (only with x_kind: energy)
<run_dir>/data/data_report.md      summary and centrality coverage
<run_dir>/data/data_overview.png   events per centrality class (new vs v6), N_ch(S), [pT](S)
```

`idx` (0 … N−1) is the event number used by every later step. S is summed in double precision from
the raw values, before the grids are stored as float32. `centrality` is the minimum-bias centrality
from S ([01_background.md](01_background.md) §6, `gouai_train/centrality.py`).
