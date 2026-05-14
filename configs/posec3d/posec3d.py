_base_ = '../../.github/pose-c3d/configs/posec3d/slowonly_r50_ntu60_xsub/joint.py'

load_from = r'../../models/posec3d_r50_ntu60.pth'
work_dir = r'../../data/output/posec3d'

# Khi co dataset ADL rieng:
# 1) sua num_classes
# 2) sua ann_file cho train/val/test
# 3) sua label map neu can

model = dict(
    cls_head=dict(
        num_classes=60
    )
)
