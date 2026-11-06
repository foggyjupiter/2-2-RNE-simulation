#!/bin/bash
# Hypothesis: the crank drive's 2nd harmonic synchronises sparse beds (N 10-35 gap).
cd /home/user/2-2-RNE-simulation
for lam in 0.15 0.3; do for n in 5 20 50 100; do echo "$n $lam"; done; done | xargs -P 1 -L 1 nice -n 19 bash -c 'gransim run configs/experiment_replica.yaml --out runs/e_probe/K$1_N$0 --set particles.count=$0 --set contact.model=hertz --set excitation.rpm=372 --set excitation.crank_ratio=$1 --set contact.wall_roughness_deg=0.5 --set contact.restitution_pp=0.95 --set contact.restitution_pw=0.7 --set time.record_time=5.3 > runs/e_probe/K$1_N$0.log 2>&1'
echo done > runs/e_probe/ALL_DONE_CRANK
