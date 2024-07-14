import argparse
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as v2
from   models import DeTr


parser = argparse.ArgumentParser()
parser.add_argument("--nepochs",      type=int,   default=100,    help="Number of epochs")
parser.add_argument("--batchsize",    type=int,   default=32,     help="Batch size")
parser.add_argument("--nwarmup",      type=int,   default=1000,   help="number of warmup steps")
parser.add_argument("--lr",           type=float, default=0.001,  help="Initial learning rate")
parser.add_argument("--trainRoot",    type=str,   required=True,  help="Root folder of training directory")
parser.add_argument("--trainAnn",     type=str,   required=True,  help="Training annotations file")
parser.add_argument("--valRoot",      type=str,   required=True,  help="Root folder of validation directory")
parser.add_argument("--valAnn",       type=str,   required=True,  help="Validation annotations file")
parser.add_argument("--nworkers",     type=int,   default=0,      help="Number of data workers. If 0, set to mp.cpu_count()/2")
args = parser.parse_args()


class CocoWrapper(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transforms=[]):
        super().__init__()
        transforms   = v2.Compose([v2.ToImage(), *transforms, v2.ToDtype(torch.float32, scale=True)])
        dataset      = torchvision.datasets.CocoDetection(root, annFile, transforms=transforms)
        cat_ids      = dataset.coco.getCatIds()
        cats         = dataset.coco.loadCats(cat_ids)
        self.names   = [cat["name"] for cat in cats]
        self.ids     = {cat: id for id, cat in enumerate(cat_ids)}
        self.dataset = torchvision.datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=["boxes", "labels"])
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        img, target = self.dataset[index]
        classes     = torch.tensor([self.ids[i.item()] for i in target['labels']]).unsqueeze(-1) if 'labels' in target else torch.zeros(0,1)
        boxes       = target['boxes'] if 'boxes' in target else torch.zeros(0,4)
        target      = torch.cat([boxes,classes], -1)
        return img, target


def CocoCollator(batch):
    imgs, targets   = zip(*batch)
    N               = max(t.shape[0] for t in targets)
    targets         = [F.pad(t, (0,0,0,N-t.shape[0]), value=-1) for t in targets]
    H               = max(x.shape[1] for x in imgs)
    W               = max(x.shape[2] for x in imgs)
    imgs            = [F.pad(x, (0,W-x.shape[2],0,H-x.shape[1]), value=0) for x in imgs]
    imgs            = torch.stack(imgs, 0)
    targets         = torch.stack(targets, 0)
    return imgs, targets


def createOptimizer(self: torch.nn.Module, momentum=0.9, lr=0.001, decay=0.0001):
    param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params    = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params  = [p for n, p in param_dict.items() if p.dim() < 2]
    assert len(decay_params) + len(nodecay_params) == len(list(filter(lambda p: p.requires_grad, self.parameters()))), "bad split"
    optim_groups = [
        {'params': decay_params,   'weight_decay': decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params    = sum(p.numel() for p in decay_params)
    num_nodecay_params  = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors       : {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors   : {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(momentum, 0.999), fused=True)
    return optimizer


valset      = CocoWrapper(args.valRoot,   args.valAnn,   transforms=[v2.Resize((640,640), antialias=True)])
nclasses    = len(valset.names)
valLoader   = torch.utils.data.DataLoader(valset, batch_size=args.batchsize, collate_fn=CocoCollator, num_workers=6)

net = DeTr(nclasses)

for i, (batch, targets) in enumerate(valLoader):
    preds, loss = net(batch, targets)
    exit(0)