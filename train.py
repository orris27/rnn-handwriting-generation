#import tensorflow as tf
import numpy as np
import torch
import model as m
from config import *

from utils import DataLoader, draw_strokes_random_color

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_loader = DataLoader(args.batch_size, args.T, args.data_scale,
                         chars=args.chars, points_per_char=args.points_per_char)
args.U = data_loader.max_U
args.c_dimension = len(data_loader.chars) + 1
args.action = 'train'

model = m.Model(args).to(device)
for e in range(args.num_epochs):
    print("epoch %d" % e)
    data_loader.reset_batch_pointer()
    for b in range(data_loader.num_batches):
        x, y, c_vec, c = data_loader.next_batch()
        if b % 100 == 0:
            print('batches %d'%(b), end=':')
            model.fit(x, y)
    torch.save(model.state_dict(), args.model_path)
