#!/bin/bash
# Pack / unpack the raw per-cell JSON files of the re-tuned judgment-unit run (they are git-ignored; the .tar.xz archives are committed).
# usage: judgment_unit_pack.sh pack|unpack     (run from this folder)
cd "$(dirname "$0")/results/judgment_unit_study_retuned" || exit 1
for r in A_groups B_groups; do
  if [ "$1" = pack ]; then tar -cJf cells_$r.tar.xz $r/single $r/sequential; else tar -xJf cells_$r.tar.xz; fi
done
