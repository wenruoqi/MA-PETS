from mbexp import mbexp
import argparse
parser = argparse.ArgumentParser() 
parser.add_argument('-env', type=str, required=True, 
                    help='Environment name: select from [smartscavs_v1]')
parser.add_argument('-ca', '--ctrl_arg', action='append', nargs=2, default=[],help='Controller arguments')
parser.add_argument('-o', '--override', action='append', nargs=2, default=[], help='Override default parameters')
parser.add_argument('-logdir', type=str, default='log',help='Directory to which results will be logged (default: ./log)')
args = parser.parse_args()

mbexp(args.env, "MPC", args.ctrl_arg, args.override, args.logdir)