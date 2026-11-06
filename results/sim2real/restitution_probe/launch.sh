#!/bin/bash
# Hypothesis: the real BB bed at high N is more gas-like (more elastic) than e = 0.8 gives.
cd /home/user/2-2-RNE-simulation
for e in 0.9 0.95; do for n in 50 100; do echo "$n $e"; done; done | xargs -P 1 -L 1 nice -n 19 bash -c 'gransim run configs/experiment_replica.yaml --out runs/e_probe/E$1_N$0 --set particles.count=$0 --set contact.model=hertz --set excitation.rpm=372 --set contact.wall_roughness_deg=0.5 --set contact.restitution_pp=$1 --set contact.restitution_pw=$1 --set time.record_time=5.3 > runs/e_probe/E$1_N$0.log 2>&1'
echo done > runs/e_probe/ALL_DONE
