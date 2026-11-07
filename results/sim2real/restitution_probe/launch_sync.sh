#!/bin/bash
# 남은 실측 편향(N 10~35에서 실측이 더 동기화)의 동역학 후보: 벽 요철, 벽·입자 반발, 마찰.
# 기준 = 보정값 e_pp 0.95, e_pw 0.7, 요철 0.5°, μ 0.3 (실행 이름 S0.95_0.7 은 runs/e_probe 에 N 5/20/50/100).
cd /home/user/2-2-RNE-simulation
B="configs/experiment_replica.yaml --set contact.model=hertz --set excitation.rpm=372 --set time.record_time=5.3"
{
for n in 10 20 35 100; do
  echo "base_N$n|--set contact.restitution_pp=0.95 --set contact.restitution_pw=0.7 --set contact.wall_roughness_deg=0.5|$n"
  echo "r0_N$n|--set contact.restitution_pp=0.95 --set contact.restitution_pw=0.7 --set contact.wall_roughness_deg=0.0|$n"
  echo "r0.2_N$n|--set contact.restitution_pp=0.95 --set contact.restitution_pw=0.7 --set contact.wall_roughness_deg=0.2|$n"
  echo "pw0.5_N$n|--set contact.restitution_pp=0.95 --set contact.restitution_pw=0.5 --set contact.wall_roughness_deg=0.5|$n"
  echo "pp0.8_N$n|--set contact.restitution_pp=0.8 --set contact.restitution_pw=0.7 --set contact.wall_roughness_deg=0.5|$n"
  echo "mu0.5_N$n|--set contact.restitution_pp=0.95 --set contact.restitution_pw=0.7 --set contact.wall_roughness_deg=0.5 --set contact.friction_pp=0.5 --set contact.friction_pw=0.5|$n"
done
} | while IFS='|' read name opts n; do echo "$name $n $opts"; done > runs/sync_probe/jobs.txt
cat runs/sync_probe/jobs.txt | xargs -P 4 -L 1 bash -c 'name=$0; n=$1; shift 1; [ -f runs/sync_probe/$name/summary.json ] || gransim run '"$B"' --out runs/sync_probe/$name --set particles.count=$n "$@" > runs/sync_probe/$name.log 2>&1'
echo done > runs/sync_probe/ALL_DONE
