import argparse
import statistics
import numpy as np 
import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric_signed_directed.data import load_directed_real_data 
from torch_geometric.seed import seed_everything

from torch_geometric.utils import add_self_loops, remove_self_loops

from src.model import GCN, GCN_FLEPE, GAT, GAT_FLEPE, GT, GT_FLEPE

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora_ml', help='Cora, Citeseer, cora_ml, telegram, Cornell, Wisconsin, Texas')
parser.add_argument('--hidden_channels', type=int, default=16)
parser.add_argument('--k', type=int, default=5)
parser.add_argument('--model', type=str, default='gcn')
parser.add_argument('--flow_method', type=str, default='pre')
parser.add_argument('--vertex_import', type=str, default='True')
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--seed',type=int, default=0)
args = parser.parse_args()
print(args)

seed_everything(args.seed)

device = torch.device('cpu')

if args.dataset in ['cora_ml', 'telegram']:
    data = load_directed_real_data(dataset=args.dataset, root='data/' + args.dataset)

elif args.dataset in ['Cornell', 'Wisconsin', 'Texas']:
    data = load_directed_real_data(dataset='WebKB', root='data/', name=args.dataset)

data = data.to(device)
num_classes = (data.y.max() - data.y.min() + 1).cpu().numpy()


if 'flepe' in args.model:
    flepe = torch.load(f'PE_data/FLEPE/{args.dataset.lower()}_{args.vertex_import.lower()}_{args.flow_method}.pt')
    flepe = flepe[:,:args.k]
    edge_dim = flepe.size(1)
else:
    flepe = None



def train(train_mask, flepe=None):
    model.train()
    optimizer.zero_grad()
    if flepe is not None:
        out = model(data.x, data.edge_index, data.edge_attr, flepe)
    else:
        out = model(data.x, data.edge_index, data.edge_attr)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def test(train_mask, val_mask, test_mask, flepe=None):
    model.eval()
    if flepe is not None:
        pred = model(data.x, data.edge_index, data.edge_attr,flepe).argmax(dim=-1)
    else:
        pred = model(data.x, data.edge_index, data.edge_attr).argmax(dim=-1)
    accs = []
    for mask in [train_mask, val_mask, test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs

acc = []
for split in range(data.train_mask.shape[1]):
    if args.model == 'gcn':
        model = GCN(in_channels=data.x.shape[1],
                    hidden_channels=args.hidden_channels,
                    out_channels=num_classes).to(device)
    elif args.model == 'gcn-flepe':
        model = GCN_FLEPE(in_channels=data.x.shape[1],
                    hidden_channels=args.hidden_channels,
                    out_channels=num_classes,
                    edge_dim=edge_dim).to(device)
        if split == 0:
            _, flepe = add_self_loops(data.edge_index, flepe)
    elif args.model == 'gat':
        model = GAT(in_channels=data.x.shape[1],
                    hidden_channels=args.hidden_channels,
                    out_channels=num_classes).to(device)
    elif args.model == 'gat-flepe':
        model = GAT_FLEPE(in_channels=data.x.shape[1],
                    hidden_channels=args.hidden_channels,
                    out_channels=num_classes,
                    edge_dim=edge_dim).to(device)
        data.edge_index,_ = remove_self_loops(data.edge_index)
    elif args.model == 'gt':
        model = GT(in_channels=data.x.shape[1],
                    hidden_channels=args.hidden_channels,
                    out_channels=num_classes).to(device)
    elif args.model == 'gt-flepe':
        model = GT_FLEPE(in_channels=data.x.shape[1],
                    hidden_channels=args.hidden_channels,
                    out_channels=num_classes,
                    edge_dim=edge_dim).to(device)
        data.edge_index,_ = remove_self_loops(data.edge_index)

#     optimizer = torch.optim.Adam([
#     dict(params=model.conv1.parameters(), weight_decay=5e-4),
#     dict(params=model.conv2.parameters(), weight_decay=0)
# ], lr=args.lr)  # Only perform weight-decay on first convolution.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=5e-4)
    train_mask = data.train_mask[:,split]
    val_mask = data.val_mask[:,split]
    test_mask = data.test_mask[:,split]

    best_val_acc = test_acc = 0
    for epoch in range(1, args.epochs + 1):
        # print(flepe.size(), data.edge_index.size())
        loss = train(train_mask, flepe)
        train_acc, val_acc, tmp_test_acc = test(train_mask, val_mask, test_mask, flepe)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
    acc.append(test_acc)
    print(f'Split: {split:02d}, Test_Acc: {test_acc:.4f}')

np.savetxt(f"{args.dataset}/{args.model}/{args.flow_method}/{args.vertex_import}/{args.k}_seed_{args.seed}.txt", np.array(acc),fmt="%.6f")
print(f'Avg ACC : {statistics.mean(acc)*100}, STD : {statistics.stdev(acc)*100}')

