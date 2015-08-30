#!/usr/bin/env python
import argparse
from main import create_rnn, train_rnn, separate

# Create the main parser
parser = argparse.ArgumentParser(description='Bat Signal Separator')

# Subparsers for create, train and separate actions
subparsers = parser.add_subparsers(help="commands")

# for Create action type
create_parser = subparsers.add_parser('create', help="Creating new training file.")
create_parser.add_argument('-f', '--filename', action="store", dest="training_file", default=None, help="Name of file you want to work on.")
create_parser.add_argument('-ni', '--inputs', action="store", dest="no_inputs", default=None, type=int, help="Number of inputs for the new training file.")
create_parser.add_argument('-no', '--outputs', action="store", dest="no_outputs", default=None, type=int, help="Number of outputs for the new training file.")
create_parser.add_argument('-nh', '--hiddens', action="store", dest="no_hiddens", default=None, type=int, help="Number of hidden neurons for the new training file.")
create_parser.set_defaults(which='create')

# for Train action type
train_parser = subparsers.add_parser('train', help="Train the given file.")
train_parser.add_argument('-f', '--filename', action="store", dest="training_file", default=None, help="Name of training file.")
train_parser.add_argument('-i', '--inputs', action="store", dest="inputs", nargs='+', default=None, help="Name of input file.")
train_parser.add_argument('-o', '--outputs', action="store", dest="outputs", nargs='+', default=None, help="Name of output file.")
train_parser.add_argument('-r', '--rate', action="store", dest="training_rate", type=float, default=0.05, help="Tell us the rate of training.")
train_parser.add_argument('-I', '--iterations', action="store", dest="no_iterations", type=int, default=1, help="Total number of iterations you want to perform.")
train_parser.add_argument('-s', '--samples', action="store", dest="no_samples", type=int, default=2000, help="Total number of samples.")
train_parser.add_argument('-O', '--offset', action="store", dest="offset", type=int, default=0, help="Offset of sound samples from beginning to use for training.")
train_parser.add_argument('-F', '--frames', action="store", dest="frames", type=int, default=1, help="Number of frames each consisting of 'n' samples.")
train_parser.set_defaults(which='train')

# for Separate action type
separate_parser = subparsers.add_parser('separate', help="Separate signals.")
separate_parser.add_argument('-f', '--fiename', action="store", dest="training_file", default=None, help="Name of training file.")
separate_parser.add_argument('-i', '--inputs', action="store", dest="inputs", nargs='+', default=None, help="Name of input files.")
separate_parser.add_argument('-o', '--outputs', action="store", dest="outputs", nargs='+', default=None, help="Name of output files.")
separate_parser.add_argument('-e', '--extras', action="store", dest="extras", default=None, help="Name of output files.")
separate_parser.set_defaults(which='separate')
args = parser.parse_args()

# function to check if any of the arguments required is absent 
def check_for_null(arguments):
    for x in arguments:
        if x is None:
            print("You haven't specified the required argument. Type -h or --help for help.")
            return True
    return False

if "which" in args:
    if args.which == 'create':
        # list of all the arguments in create
        arguments = [args.training_file, args.no_inputs, args.no_outputs, args.no_hiddens]
        if not check_for_null(arguments):
            create_rnn(args.training_file, args.no_inputs, args.no_outputs, args.no_hiddens)

    elif args.which == 'train':
        # list of all the arguments in train
        arguments = [args.training_file, args.inputs, args.outputs, 
                args.training_rate, args.no_iterations, args.no_samples, args.offset, args.frames]
        if not check_for_null(arguments[:3]):
            train_rnn(args.training_file, args.inputs, args.outputs, 
                    args.training_rate, args.no_iterations, args.no_samples, args.offset, args.frames)

    elif args.which == 'separate':
        # list of all the arguments in separate
        arguments = [args.training_file, args.inputs, args.outputs, args.extras]
        if not check_for_null(arguments[:3]):
            separate_rnn(args.training_file, args.inputs, args.outputs, args.extras)
else:
    print("You haven't specified any command. Type -h or --help for help.")
