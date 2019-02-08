from model import *
from trainer import Trainer
import argparse

fe = FrontEnd()
d = D()
q = Q()
g = G()

for i in [fe, d, q, g]:
  i.cuda()
  i.apply(weights_init)

def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--image_size', type=int, default=28, help='The side length N to convert images to NxN.')
    parser.add_argument('--conv_dim', type=int, default=128)
    parser.add_argument('--noise_size', type=int, default=74)
    parser.add_argument('--cont_dim_size', type=int, default=2)
    parser.add_argument('--cat_dim_size', type=int, default=10)

    # Training hyper-parameters
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16, help='The number of images in a batch.')
    parser.add_argument('--num_workers', type=int, default=0, help='The number of threads to use for the DataLoader.')
    parser.add_argument('--lr', type=float, default=0.0003, help='The learning rate (default 0.0003)')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.99)

    # Data sources
    parser.add_argument('--emoji', type=str, default='Apple', choices=['Apple', 'Facebook', 'Windows'], help='Choose the type of emojis to generate.')

    # Directories and checkpoint/sample iterations
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_vanilla')
    parser.add_argument('--sample_dir', type=str, default='./samples_vanilla')
    parser.add_argument('--log_step', type=int , default=10)
    parser.add_argument('--sample_every', type=int , default=200)
    parser.add_argument('--checkpoint_every', type=int , default=400)

    return parser




parser = create_parser()
opts = parser.parse_args()

print("getting trainer")
trainer = Trainer(g, fe, d, q)
print("Start training")
trainer.train(opts)