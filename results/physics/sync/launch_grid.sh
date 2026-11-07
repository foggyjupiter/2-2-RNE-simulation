#!/bin/bash
# 벽 반발계수 e_pw × 개수 N 격자 (BB, 372 rpm, e_pp 0.95, 요철 0.5°). 기존 sync_probe 실행(base=pw0.7, pw0.4~0.6)은 재사용.
cd /home/user/2-2-RNE-simulation
B="configs/experiment_replica.yaml --set contact.model=hertz --set excitation.rpm=372 --set time.record_time=5.3"
name_of() { [ "$1" = 0.7 ] && echo base || echo "pw$1"; }
{
for e in 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
  for n in 1 2 5 10 20 35 50 100; do
    echo "$(name_of $e)_N$n $n --set contact.restitution_pp=0.95 --set contact.restitution_pw=$e --set contact.wall_roughness_deg=0.5"
  done
done
} > runs/sync_probe/jobs_grid.txt
cat runs/sync_probe/jobs_grid.txt | xargs -P 4 -L 1 bash -c 'name=$0; n=$1; shift 1; [ -f runs/sync_probe/$name/summary.json ] || gransim run '"$B"' --out runs/sync_probe/$name --set particles.count=$n "$@" > runs/sync_probe/$name.log 2>&1'
echo done > runs/sync_probe/ALL_DONE_GRID
