#!/bin/bash
# launch_sync.sh 후속: 벽 반발계수 0.4 / 0.6 (N 10/20/35/100)과 0.5의 N 5/50.
cd /home/user/2-2-RNE-simulation
B="configs/experiment_replica.yaml --set contact.model=hertz --set excitation.rpm=372 --set time.record_time=5.3"
{
for pw in 0.4 0.6; do for n in 10 20 35 100; do
  echo "pw${pw}_N$n $n --set contact.restitution_pp=0.95 --set contact.restitution_pw=$pw --set contact.wall_roughness_deg=0.5"
done; done
for n in 5 50; do echo "pw0.5_N$n $n --set contact.restitution_pp=0.95 --set contact.restitution_pw=0.5 --set contact.wall_roughness_deg=0.5"; done
for n in 5 50; do echo "base_N$n $n --set contact.restitution_pp=0.95 --set contact.restitution_pw=0.7 --set contact.wall_roughness_deg=0.5"; done
} > runs/sync_probe/jobs2.txt
cat runs/sync_probe/jobs2.txt | xargs -P 4 -L 1 bash -c 'name=$0; n=$1; shift 1; [ -f runs/sync_probe/$name/summary.json ] || gransim run '"$B"' --out runs/sync_probe/$name --set particles.count=$n "$@" > runs/sync_probe/$name.log 2>&1'
echo done > runs/sync_probe/ALL_DONE2
