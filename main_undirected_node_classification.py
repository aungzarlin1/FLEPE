import argparse

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.logging import log
from torch_geometric.utils import add_self_loops
from torch_geometric.seed import seed_everything

from torch_geometric_signed_directed.data import load_directed_real_data 

from src.model import GCN, GCN_FLEPE, GAT, GAT_FLEPE, GT, GT_FLEPE

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora_ml', help='Cora, Citeseer, cora_ml, telegram, Cornell, Wisconsin, Texas')
parser.add_argument('--hidden_channels', type=int, default=16)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--model', type=str, default='gcn')
parser.add_argument('--vertex_import', type=bool, default=True)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed',type=int, default=0)
args = parser.parse_args()

seed_everything(args.seed)

device = torch.device('cpu')

if args.dataset in ['Cora', 'Citeseer']:
    dataset = Planetoid(root='data', name=args.dataset)
    data = dataset[0]

data = data.to(device)

if 'flepe' in args.model:
    flepe = torch.load(f'PE_data/FLEPE/{args.dataset.lower()}_{str(args.vertex_import).lower()}.pt')
    flepe = flepe[:,:args.k]
    edge_dim = flepe.size(1)
else:
    flepe = None

if args.model == 'gcn':
    model = GCN(in_channels=dataset.num_features,
                hidden_channels=args.hidden_channels,
                out_channels=dataset.num_classes).to(device)
elif args.model == 'gcn-flepe':
    model = GCN_FLEPE(in_channels=dataset.num_features,
                hidden_channels=args.hidden_channels,
                out_channels=dataset.num_classes,
                edge_dim=edge_dim).to(device)
    _, flepe = add_self_loops(data.edge_index, flepe)
elif args.model == 'gat':
    model = GAT(in_channels=dataset.num_features,
                hidden_channels=args.hidden_channels,
                out_channels=dataset.num_classes).to(device)
elif args.model == 'gat-flepe':
    model = GAT_FLEPE(in_channels=dataset.num_features,
                hidden_channels=args.hidden_channels,
                out_channels=dataset.num_classes,
                edge_dim=edge_dim).to(device)
elif args.model == 'gt':
    model = GT(in_channels=dataset.num_features,
                hidden_channels=args.hidden_channels,
                out_channels=dataset.num_classes).to(device)
elif args.model == 'gt-flepe':
    model = GT_FLEPE(in_channels=dataset.num_features,
                hidden_channels=args.hidden_channels,
                out_channels=dataset.num_classes,
                edge_dim=edge_dim).to(device)


# optimizer = torch.optim.Adam([
#     dict(params=model.conv1.parameters(), weight_decay=5e-4),
#     dict(params=model.conv2.parameters(), weight_decay=0)
# ], lr=args.lr)  # Only perform weight-decay on first convolution.

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

def train(flepe=None):
    model.train()
    optimizer.zero_grad()
    if flepe is not None:
        out = model(data.x, data.edge_index, data.edge_attr, flepe)
    else:
        out = model(data.x, data.edge_index, data.edge_attr)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def test(flepe=None):
    model.eval()
    if flepe is not None:
        pred = model(data.x, data.edge_index, data.edge_attr, flepe).argmax(dim=-1)
    else:
        pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs


best_val_acc = test_acc = 0
times = []
for epoch in range(1, args.epochs + 1):
    loss = train(flepe)
    train_acc, val_acc, tmp_test_acc = test(flepe)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc)
    
