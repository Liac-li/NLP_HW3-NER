import argparse
from numpy import argmax
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import tqdm
import pandas as pd

from dataProc import NERDataSet, NERDSWrapper 
from model import BiRNN_NER
from colorUtil import set_color, COLOR


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
    parser.add_argument('--just_eval', action='store_true')

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
    test_dl = DataLoader(test_wrapper, batch_size=args.batch_size, shuffle=False)

    # Build Model
    # ! Increase weight of art, net eve
    class_weight = torch.ones(ds_ner.target_size)
    class_weight[[8, 9, 12, 13, 14, 15]] = 4 

    model = BiRNN_NER(args, ds_ner.target_size, class_weight=class_weight)
    if args.just_eval:
        model.load_state_dict(SLModel(None, args.save_dir, 9, mode='load'))
    optimizer =  optim.Adam(model.parameters(), lr=args.lr)
    
    # Train
    model.to(DEVICE)
    best_val = 1e4 

    if not args.just_eval:
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
    
    # Predict Test set
    bar = tqdm.tqdm(test_dl)
    test_tags = []
    for bi, (x, mask) in enumerate(bar):
        tags = model.forward(x.to(DEVICE), mask.to(DEVICE)).cpu().detach().numpy()
        cut_indexes = mask.cpu().detach().numpy() 

        for ni, tags in enumerate(tags):
            sentence_tag = []
            for pos, items in enumerate(tags):
                if pos <=cut_indexes[ni]:
                    sentence_tag.append(argmax(items))
            sentence_tag = map(lambda x: ds_ner.id2tag[x], sentence_tag) 

            test_tags.append(' '.join(sentence_tag))

    ids = [i for i in range(len(test_tags))]
    df = pd.DataFrame({'id': ids, 'Tag': test_tags})
    df.to_csv('res.csv', index=False)

    print(set_color(f"Good Job", COLOR.YELLOW))

    
