#!/bin/bash
# 상자 7×7×5 데이터: 몫이 모두 끝나면 목록(manifest/metadata)만 받아 전체 목록을 만들어 올림.
D=/home/user/2-2-RNE-simulation/datasets
export HF_TOKEN=proxy-injected
cd /home/user/2-2-RNE-simulation
merge() {  # prefix n local spec total_runs
  mkdir -p $D/$3 && cp configs/box7/$4 $D/$3/sweep_spec.yaml
  python3 $D/wait_markers.py $1 $2 120 $5 > $D/$3.merge.log 2>&1
  gransim upload $D/$3 --repo FoggyJupiter/rne-granular-acoustics --prefix $1 --fetch-meta >> $D/$3.merge.log 2>&1
  echo "MERGED $1 $(date -u +%H:%M)" | tee -a $D/box7_merge_status.log
}
merge box7/v3_nosig 12 box7_main sweep_main.yaml 1200 &
merge box7/multi_rpm_nosig 15 box7_mr sweep_multi_rpm.yaml 1800 &
merge box7/bb_fixed 4 box7_bb sweep_bb_fixed.yaml 300 &
merge box7/lowe 3 box7_lowe sweep_lowe.yaml 300 &
wait
echo "ALL BOX7 MERGED $(date -u +%H:%M)" | tee -a $D/box7_merge_status.log
