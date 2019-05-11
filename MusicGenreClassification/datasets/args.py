from args import parser


sub_parser = parser.add_argument_group(
    'datasets', 'Parameters for datasets.')
sub_parser.add_argument('--sample_rate', type=int, default=22050)
sub_parser.add_argument('--num_target_samples', type=int, default=660980)
sub_parser.add_argument('--num_target_segment_samples', type=int, default=66560)  # 66098 # each segment is about 3 seconds
sub_parser.add_argument('--n_fft', type=int, default=2048)
sub_parser.add_argument('--hop_length', type=int, default=1024)
sub_parser.add_argument('--pad_signal', type=bool, default=False)


def build_args():
    args = vars(parser.parse_args())
    return args