#!/usr/bin/env python

import argparse
from main import create_network, train_network, separate


# Create the main parser
parser = argparse.ArgumentParser(description='Bat Signal Separator')

# Subparsers for create, train and separate actions
subparsers = parser.add_subparsers(help="commands")

# for Create action type
create_parser = subparsers.add_parser('create', help="Create new training file.")
create_parser.add_argument('-f', '--filename', action="store", dest="training_file", default=None, help="Name of file you want to save new neural network.")
create_parser.add_argument('-i', '--inputs', action="store", dest="num_inputs", default=None, type=int, help="Number of inputs of the neural network.")
create_parser.add_argument('-o', '--outputs', action="store", dest="num_outputs", default=None, type=int, help="Number of outputs for the neural network.")
create_parser.add_argument('-s', '--samples', action="store", dest="num_samples", default=None, type=int, help="Number of samples processed once by the neural network.")
create_parser.add_argument('-hl', '--hidden_layers', action="store", dest="num_hidden_layers", default=None, type=int, help="Number of hidden layer neurons for the neural network.")
create_parser.set_defaults(which='create')

# for Train action type
train_parser = subparsers.add_parser('train', help="Train the given file.")
train_parser.add_argument('-f', '--filename', action="store", dest="training_file", default=None, help="Name of the file where the neural network is saved.")
train_parser.add_argument('-i', '--inputs', action="store", dest="inputs", nargs='+', default=None, help="Names of the input sound files.")
train_parser.add_argument('-o', '--outputs', action="store", dest="outputs", nargs='+', default=None, help="Names of the output sound files.")
train_parser.add_argument('-r', '--rate', action="store", dest="learning_rate", type=float, default=0.03, help="Learning rate to use during training.")
train_parser.add_argument('-I', '--iterations', action="store", dest="num_iterations", type=int, default=30, help="Total number of iterations to perform during training.")
train_parser.add_argument('-O', '--offset', action="store", dest="offset", type=int, default=0, help="Offset of sound samples from beginning to use for training.")
train_parser.add_argument('-F', '--frames', action="store", dest="frames", type=int, default=None, help="Number of frames each consisting of given number of samples to use for training.")
train_parser.set_defaults(which='train')

# for Separate action type
separate_parser = subparsers.add_parser('separate', help="Separate signals.")
separate_parser.add_argument('-f', '--filename', action="store", dest="training_file", default=None, help="Name of the file where the neural network is saved.")
separate_parser.add_argument('-i', '--inputs', action="store", dest="inputs", nargs='+', default=None, help="Name of the input sound files.")
separate_parser.add_argument('-o', '--outputs', action="store", dest="outputs", nargs='+', default=None, help="Name of the output sound files.")
separate_parser.add_argument('-e', '--extras', action="store", dest="extras", default=None, help="Name of extra sound files to plot.")
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
        arguments = [args.training_file, args.num_samples, args.num_inputs, args.num_outputs, args.num_hidden_layers]
        if not check_for_null(arguments):
            create_network(args.training_file, args.num_samples, args.num_inputs, args.num_outputs, args.num_hidden_layers)

    elif args.which == 'train':
        # list of all the arguments in train
        arguments = [args.training_file, args.inputs, args.outputs]
        if not check_for_null(arguments[:3]):
            train_network(args.training_file, args.inputs, args.outputs, 
                    args.learning_rate, args.num_iterations, args.offset, args.frames)

    elif args.which == 'separate':
        # list of all the arguments in separate
        arguments = [args.training_file, args.inputs, args.outputs]
        if not check_for_null(arguments[:3]):
            separate(args.training_file, args.inputs, args.outputs, args.extras)
else:
    print("You haven't specified any command. Type -h or --help for help.")
