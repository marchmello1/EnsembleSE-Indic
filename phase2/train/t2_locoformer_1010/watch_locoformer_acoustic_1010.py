"""Evaluate the acoustic1600 run with the existing frozen PCM/ASR gate."""
from pathlib import Path

source = Path('/mnt/nvme_disk1/t2_locoformer_1010/watch_locoformer_rnnt_1010.py').read_text()
source = source.replace('locoformer_rnnt_1010_s800.pt', 'locoformer_acoustic_1010_s1600.pt')
source = source.replace('rnnt800', 'acoustic1600')
exec(compile(source, __file__, 'exec'))
