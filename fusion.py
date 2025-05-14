from mmcv import load, dump
from atemgcn.smp import *


joint = load('work_dirs/ntu60_xsub/j/result.pkl')
bone = load('work_dirs/ntu60_xsub/b/result.pkl')
joint_motion = load('work_dirs/ntu60_xsub/jm/result.pkl')
bone_motion = load('work_dirs/ntu60_xsub/bm/result.pkl')

label = load_label('data/ntu60.pkl', 'xsub_val')

# J+B Fusion result
fused = comb([joint, bone], [1, 1])
print('Top-1', top1(fused, label))

# J+B+JM+BM Fusion result
fused = comb([joint, bone, joint_motion, bone_motion], [2, 2, 1, 1])
print('Top-1', top1(fused, label))
