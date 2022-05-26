from html.entities import entitydefs
import torch
import numpy as np
import threading

def f1_score(y_pred:torch.Tensor, y_true:torch.Tensor, outer=0):
    """ Compute F1 score
    
        F1 = 2 * (precision * recall) / (precision + recall)
        
        Parameters:
            y_pred: torch.Tensor, shape = (batch_size, seq_len, c)
            y_true: torch.Tensor, shape = (batch_size, seq_len)
            outer: the O tag in NER

        Exactly matched entities are correct (b, i, entity)            
    """
    y_pred = y_pred.cpu().data.numpy()
    # y_pred = np.argmax(y_pred, axis=-1) # (batch_size, seq_len)
    y_true = y_true.cpu().data.numpy()
    assert y_pred.shape == y_true.shape
    
    # Compute precision and recall
    batch = y_pred.shape[0]
    thread_num = 3
    y_preds = np.array_split(y_pred, thread_num)
    y_trues = np.array_split(y_true, thread_num)

    result = {}
    threads = []
    for i in range(thread_num):
        args = (y_preds[i], y_trues[i], result, i)
        threads.append(threading.Thread(target=_multi_f1, args=args))

    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
    total_s = 0
    total_g = 0
    total_corct = 0
    for id, items in result.items():
        total_s += items[0]
        total_g += items[1]
        total_corct += items[2]
        
        
    print(total_g, total_s, total_corct)

    percision = total_corct / total_s if total_s > 0 else 0
    recall = total_corct / total_g if total_g > 0 else 0
    f1 = 2 * percision * recall / (percision + recall) if (percision + recall) > 0 else 0
    return f1

    
def _multi_f1(y_pred, y_true, results, thread_id):
    is_begin = lambda x: x % 2 != 0
    print(y_pred, y_true)
    
    def sen_count(y_pred, y_true):
        s = 0; g = 0; correct = 0 
        matching = False
        last_entity = None
        for pos, entity in enumerate(y_true):
            last_entity = entity if last_entity is None else last_entity
            if is_begin(entity):
                s += 1
            if is_begin(y_pred[pos]):
                g += 1

            # matching ?

            if entity != y_pred[pos] or entity != last_entity:
                if matching:
                    matching = False
                    correct += 1
            else:
                matching = False

            matching = True if is_begin(entity) and entity == y_pred[pos] else matching
                
        return s, g, correct
    
    s = 0; g = 0 
    correct = 0

    for b, batch in enumerate(y_true):
        sen_s, sen_g, sen_corct = sen_count(y_pred[b], batch)
        
        s += sen_s
        g += sen_g
        correct += sen_corct

    results[thread_id] = (s, g, correct)
    
if __name__ == '__main__':
    pred = torch.Tensor([[0, 0, 1, 2, 0, 3, 5], [0, 1, 0, 0, 0, 0, 0]])
    truth = torch.Tensor([[0, 0, 1, 2, 3, 3, 0], [0, 1, 0, 0, 0, 0, 0]])

    f1_score(pred, truth)

