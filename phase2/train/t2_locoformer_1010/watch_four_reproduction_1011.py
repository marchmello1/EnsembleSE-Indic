from pathlib import Path
root=Path('/mnt/nvme_disk1/t2_locoformer_1010')
source=(root/'watch_loco_spectral_fusion_gate_1010.py').read_text()
source=source.replace("ck=root/'loco_spectral_fusion_1010_s1200.pt'",
    "ck=Path('/mnt/nvme_disk1/t2_gap_urgenet_1009/predictor_fulladapt_1010_s800.pt')")
source=source.replace('eval_loco_spectral_fusion_1010.py','reproduce_four_network_588_1011.py')
source=source.replace('loco_spectral1200','four_reproduction')
source=source.replace("CUDA_VISIBLE_DEVICES='7'","CUDA_VISIBLE_DEVICES='0'")
exec(compile(source,str(root/'watch_four_reproduction_1011.py'),'exec'))
