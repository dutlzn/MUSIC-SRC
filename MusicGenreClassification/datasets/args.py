from args import parser


sub_parser = parser.add_argument_group(
    'datasets', 'Parameters for datasets.')
# 位置参数和可选参数， 创建合适的分组

# 音频采样频率
sub_parser.add_argument('--sample_rate', type=int, default=22050)
sub_parser.add_argument('--num_target_samples', type=int, default=660980)
sub_parser.add_argument('--num_target_segment_samples', type=int, default=66560)  # 66098 # each segment is about 3 seconds
# 快速傅里叶变换窗口大小
sub_parser.add_argument('--n_fft', type=int, default=2048)
# 相邻帧之间重叠样本数量
sub_parser.add_argument('--hop_length', type=int, default=1024)
# 是否对信号进行增补使得所有帧居中对齐
sub_parser.add_argument('--pad_signal', type=bool, default=False)


def build_args():
    args = vars(parser.parse_args())
    return args

# ---------------------------
# In[1]
import argparse
parser = argparse.ArgumentParser(prog='PROG',
                                add_help=False)
group1 = parser.add_argument_group('group1', 'group1 description')
# In[2]
group1.add_argument('foo', help='foo help')]

# In[3]
parser.print_help()
# In[4]
660980/10