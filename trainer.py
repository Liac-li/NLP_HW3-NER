from mindspore import nn, Model
from mindspore.train.callback import LossMonitor, CheckpointConfig, ModelCheckpoint, TimeMonitor, Callback
from mindspore import load_checkpoint, load_param_into_net
from mindspore.nn import F1
import mindspore as ms
from utils import apply_eval, EvalCallBack
from tqdm import tqdm
import numpy as np
import os


class Trainer:

    def __init__(self, config, model, train_dataset, dev_dataset,
                 test_dataset):
        self.config = config
        # 载入数据集
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        # 载入模型，注意下面这四步的写法，最好去看一看API，与torch有所差别
        self.model = model
        
        cos_decay_lr = nn.CosineDecayLR(0.0001, 0.01, 5)
        # self.optimizer = nn.Adam(model.trainable_params(),
        #                          learning_rate=config.learning_rate)
        self.optimizer = nn.Adam(model.trainable_params(), learning_rate=cos_decay_lr)
 
        self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True,
                                                     reduction='mean')
        self.TextCNN_Model = Model(self.model, self.loss, self.optimizer,
                                   {'F1': F1()}, amp_level='O3')

        # 最佳模型保存路径
        self.best_ckpt = os.path.join(config.model_path, config.besk_ckpt_name)

        # 下面也是mindspore特有的操作，注意看怎么用的，建议看API
        self.loss_cb = LossMonitor(per_print_times=50)
        self.time_cb = TimeMonitor(data_size=train_dataset.get_dataset_size())
        eval_param_dict = {
            "model": self.TextCNN_Model,
            "dataset": self.dev_dataset,
            "metrics_name": "Acc"
        }
        self.eval_cb = EvalCallBack(
            self.config,
            apply_eval,
            eval_param_dict,
        )

    def train(self, ):
        if self.config.device_target == "CPU":
            self.TextCNN_Model.train(epoch=self.config.num_epochs,
                                     train_dataset=self.train_dataset,
                                     callbacks=[
                                         self.time_cb,
                                         self.eval_cb,
                                         self.loss_cb,
                                     ],
                                     dataset_sink_mode=False)
        else:
            self.TextCNN_Model.train(
                epoch=self.config.num_epochs,
                train_dataset=self.train_dataset,
                callbacks=[self.time_cb, self.eval_cb, self.loss_cb])

        print('testing...')
        pred_array = self.test(self.best_ckpt)
        pred_list = pred_array.tolist()
        self.to_txt(pred_list)

    def to_txt(self, pred_list):
        with open('result.txt', 'w', encoding='utf-8') as f:
            for i in range(len(pred_list)):
                f.write(str(pred_list[i]))
                f.write('\n')

    def test(self, best_ckpt):
        param_dict = load_checkpoint(
            best_ckpt)  # best_ckpt: [str] path to the best checkpoint
        load_param_into_net(self.model, param_dict)

        pred_list = np.array([], dtype=int)
        print('Testing...')
        for data in tqdm(self.test_dataset.create_dict_iterator()):
            eval_data = data['data']
            raw_label = self.TextCNN_Model.predict(eval_data)
            pred = np.argmax(raw_label, axis=1)
            pred_list = np.append(pred_list, pred)

        return pred_list
    
    def save_model(self, best_ckpt, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            
        param_dict = load_checkpoint(best_ckpt)
        load_param_into_net(self.model, param_dict)
        ms.save_checkpoint(self.model, save_path+"cnn.ckpt")
            
        
