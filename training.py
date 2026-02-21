import torch
import torch.nn as nn
import copy

from network import DSResBlock
from kernel import Functions

class Training(object):
    def __init__(self,train_loader,val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def train(self,CFG):
        model = DSResBlock.make_model(width_mult=CFG['model_width_mult'],dropout=0)
        epochs = 30
        opt = torch.optim.Adam(model.parameters(),lr=CFG['lr'],weight_decay=CFG['weight_decay'])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
        loss_fn = nn.CrossEntropyLoss()
        
        best_state = None
        best_val = -1

        for epoch in range(1,epochs+1):
            model.train()
            
            if epoch == 6:
                model.apply(DSResBlock.freeze_bn)
            
            avg_loss = 0.0
            for batch in self.train_loader:
                data = batch['x']
                target = batch['label']
                
                opt.zero_grad()
                pred = model(data)
                loss = loss_fn(pred,target)
                loss.backward()
                opt.step()
                
                avg_loss += loss.item()

            avg_loss /= len(self.train_loader)
            val_acc = Functions.eval_acc(model,self.val_loader)
            val_loss = Functions.eval_loss(model,self.val_loader,loss_fn)

            if val_acc > best_val:
                best_val = val_acc
                best_state = copy.deepcopy(model.state_dict())

            sched.step()
            
            print(f'Epoch {epoch}/{epochs}|training loss: {avg_loss:.3f}|val_acc: {val_acc:.2f}%|val_loss: {val_loss:.3f}')
            
        return model,best_state