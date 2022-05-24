from mindspore.train.callback import LossMonitor, CheckpointConfig, ModelCheckpoint, TimeMonitor, Callback
from mindspore import save_checkpoint
from tqdm import tqdm
from sklearn import metrics
import numpy as np
import os
import stat


def apply_eval(eval_param):
    eval_model = eval_param['model']
    eval_ds = eval_param['dataset']
    pred_list = np.array([], dtype=int)
    true_list = np.array([], dtype=int)
    print('Evaluating...')
    for data in tqdm(eval_ds.create_dict_iterator()):
        eval_data = data['data']
        raw_label = eval_model.predict(eval_data)
        true_label = data['label'].asnumpy()
        pred = np.argmax(raw_label, axis=1)
        pred_list = np.append(pred_list, pred)
        true_list = np.append(true_list, true_label)
    acc = metrics.accuracy_score(true_list, pred_list)
    return acc


# mindspore回调类的特有写法， 这里我在做模型验证的时候用自定义回调类，使得训练和验证都在model.train()中进行
class EvalCallBack(Callback):
    """
    回调类，获取训练过程中模型的信息
    """

    def __init__(self,
                 config,
                 eval_function,
                 eval_param_dict,
                 eval_start_epoch=0,
                 metrics_name="Acc"):
        super(EvalCallBack, self).__init__()
        self.config = config
        self.eval_param_dict = eval_param_dict
        self.eval_function = eval_function
        self.eval_start_epoch = eval_start_epoch
        interval = config.eval_interval  #每几轮进行一次验证
        if interval < 1:
            raise ValueError("interval should >= 1.")
        self.interval = interval
        self.save_best_ckpt = config.model_path
        self.best_res = 0
        self.best_epoch = 0
        if not os.path.isdir(config.model_path):
            os.makedirs(config.model_path)
        self.best_ckpt_path = os.path.join(config.model_path,
                                           config.besk_ckpt_name)
        self.metrics_name = metrics_name

    def remove_ckpoint_file(self, file_name):
        os.chmod(file_name, stat.S_IWRITE)
        os.remove(file_name)

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        loss_epoch = cb_params.net_outputs
        if cur_epoch >= self.eval_start_epoch and (
                cur_epoch - self.eval_start_epoch) % self.interval == 0:
            res = self.eval_function(self.eval_param_dict)
            print('Epoch {}/{}'.format(cur_epoch, self.config.num_epochs))
            print('-' * 10)
            print('train Loss: {}'.format(loss_epoch))
            print('val Acc: {}'.format(res))
            if res >= self.best_res:
                self.best_res = res
                self.best_epoch = cur_epoch
                if self.save_best_ckpt:
                    if os.path.exists(self.best_ckpt_path):
                        self.remove_ckpoint_file(self.best_ckpt_path)
                    save_checkpoint(cb_params.train_network,
                                    self.best_ckpt_path)

    def end(self, run_context):
        print("End training, the best {0} is: {1}, the best {0} epoch is {2}".
              format(self.metrics_name, self.best_res, self.best_epoch),
              flush=True)
