cd /mnt/nvme_disk1; export PYTHONNOUSERSITE=1 PYTHONPATH=/mnt/nvme_disk1/claude_stub
R=/mnt/nvme_disk1/official_validation_20260907; O=/mnt/nvme_disk1/claude_asrbranch_1015
for v in ens mp bs pr lo; do
  /usr/bin/python3 eval_track2_sravaani_pair.py $R $O/$v $O/wer_$v.json --all-synthetic --limit 100000 --require-extra-audio $O/$v >/dev/null 2>&1
done
/usr/bin/python3 - <<'PY'
import json
O = '/mnt/nvme_disk1/claude_asrbranch_1015'
for v in ('ens', 'mp', 'bs', 'pr', 'lo'):
    d = json.load(open(f'{O}/wer_{v}.json')); print(f"ASR {v:<4} synthetic errors {d['errors']['enhanced']}/{d['reference_words']}  (noisy {d['errors']['noisy']}, clean-ref ceiling ~510)")
PY
