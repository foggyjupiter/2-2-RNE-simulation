#!/bin/bash
cd /home/user/2-2-RNE-simulation
for n in 5 20 50 100; do echo "$n"; done | xargs -P 1 -L 1 nice -n 19 bash -c 'gransim run configs/experiment_replica.yaml --out runs/e_probe/V0.9_0.7_N$0 --set particles.count=$0 --set contact.model=hertz --set contact.restitution_law=viscoelastic --set excitation.rpm=372 --set contact.wall_roughness_deg=0.5 --set contact.restitution_pp=0.9 --set contact.restitution_pw=0.7 --set time.record_time=5.3 > runs/e_probe/V0.9_0.7_N$0.log 2>&1'
echo done > runs/e_probe/ALL_DONE_VISCO
