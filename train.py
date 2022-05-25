import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import tqdm

from dataProc import NERDataSet, NERDSWrapper 
from model import BiRNN_NER


def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='models/')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--log_dir', type=str, default='logs/')

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--save_best', action='store_true')

    parser.add_argument('--embed_size', type=int, default=500)
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--rnn_layers', type=int, default=2)

    args = parser.parse_args()
    return args


def SLModel(model, save_dir, epoch, mode='save'):
    res = None
    if mode == 'save':
        torch.save(model.state_dict(), save_dir + 'model_' + str(epoch) + '.pth')
    else:
        res = torch.load(save_dir + 'model_' + str(epoch) + '.pth')
    return res

if __name__ == '__main__':
    args = getArgs()
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data
    ds_ner = NERDataSet(args.data_dir)
    args.vocab_size = ds_ner.vocab_size # For lstm embedding
    datas = ds_ner.build_dataSets()
    train_x, train_y, train_mask, valid_x, valid_y, valid_mask, test_x, test_mask = datas
    
    train_wrapper = NERDSWrapper(train_x, train_y, train_mask) 
    valid_wrapper = NERDSWrapper(valid_x, valid_y, valid_mask)
    test_wrapper = NERDSWrapper(test_x, None, test_mask, is_test=True)
    
    train_dl = DataLoader(train_wrapper, batch_size=args.batch_size, shuffle=True)
    valid_dl = DataLoader(valid_wrapper, batch_size=args.batch_size, shuffle=False)
    test_wrapper = DataLoader(test_wrapper, batch_size=args.batch_size, shuffle=False)
    
    # Build Model
    model = BiRNN_NER(args, ds_ner.target_size)
    optimizer =  optim.Adam(model.parameters(), lr=args.lr)
    
    # Train
    model.to(DEVICE)
    best_val = 1e4 

    for epoch in range(args.epochs): 
        # Train
        model.train()
        bar = tqdm.tqdm(train_dl)
        for bi, (x, y, mask) in enumerate(bar):
            model.zero_grad()
            
            loss = model.loss(x.to(DEVICE), y.to(DEVICE), mask.to(DEVICE))
            loss.backward()
            optimizer.step()
            bar.set_description(f"{epoch}/{args.epochs} | loss: {loss.item():.4f}")

        # Validate
        model.eval()
        with torch.no_grad():
            # TODO: implement F1-Score
            ...

        # Save Model  
        res = SLModel(model, args.save_dir, epoch, mode='save')
    

    print(args)

    
