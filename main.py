import argparse
from load_data import DatasetGenerator, Preload_Data
from mindspore import context
import mindspore.dataset as ds
import mindspore as ms
from model import TextCNN, TextRNN, TextRCNN
from trainer import Trainer
import numpy as np


# 参数设置
def args_init():
    parser = argparse.ArgumentParser(description='MindSpore TextCNN Example')
    parser.add_argument('--embedding_size', type=int,
                        default=300)  #文本embedding维度
    parser.add_argument('--train_path', type=str,
                        default="data/train.txt")  #训练集路径
    parser.add_argument('--dev_path', type=str, default="data/dev.txt")  #验证集路径
    parser.add_argument('--test_path', type=str,
                        default="data/test.txt")  #测试集路径
    parser.add_argument('--vocab_path',
                        type=str,
                        default="checkpoint/vocab.pkl")
    parser.add_argument('--model_path', type=str, default="checkpoint")  #存储路径
    parser.add_argument('--besk_ckpt_name', type=str,
                        default="best.ckpt")  #最佳模型
    parser.add_argument(
        '--device_target',
        type=str,
        default="GPU",
    )  #用GPU跑
    parser.add_argument(
        '--save_best_ckpt',
        type=bool,
        default=True,
    )  #存不存最佳模型
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_classes', type=int, default=10)  #分类类别数
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=2,
    )  #训练总epoch
    parser.add_argument(
        '--batch_size',
        type=int,
        default=250,
    )
    parser.add_argument(
        '--pad_size',
        type=int,
        default=32,
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=3*1e-4,
    )  #学习率
    parser.add_argument(
        '--core_channels',
        type=int,
        default=256,
    )  #CNN输出通道数
    parser.add_argument('--save_checkpoint_steps', type=int, default=300)
    parser.add_argument('--eval_interval', type=int,
                        default=2)  #训练每2个epoch跑一次验证集

    args = parser.parse_args()
    return args


# 参数初始化
args = args_init()
# 设置mindspore的运行环境, PYNATIVE_MODE为动态图模式 GRAPH_MODE为静态图模式，设置好GPU
context.set_context(
    mode=context.GRAPH_MODE,
    device_target=args.device_target,
)
# 获得预处理的数据
p = Preload_Data(args)
train_text, train_labels, dev_text, dev_labels, test_text, vocab_size = p.build_dataset(
)
args.vocab_size = vocab_size
# 生成训练集
train_generator = DatasetGenerator(train_text, train_labels)
train_dataset = ds.GeneratorDataset(train_generator, ['data', 'label'],
                                    shuffle=True)
train_dataset = train_dataset.batch(
    batch_size=args.batch_size)  #这里使用.batch()操作来设置batch大小
# 生成验证集
dev_generator = DatasetGenerator(dev_text, dev_labels)
dev_dataset = ds.GeneratorDataset(dev_generator, ['data', 'label'],
                                  shuffle=True)
dev_dataset = dev_dataset.batch(batch_size=args.batch_size)


def generator_multidimensional(test_text):
    for i in range(len(test_text)):
        yield (np.array(test_text[i]), )


# 生成测试集
test_generator = generator_multidimensional(test_text)
test_dataset = ds.GeneratorDataset(source=test_generator,
                                   column_names=["data"])
test_dataset = test_dataset.batch(batch_size=args.batch_size)

# 模型
# model = TextCNN(config=args)
model = TextRNN(args.vocab_size, args.embedding_size, args.embedding_size, args.num_classes, num_direction=2)

# model = TextRCNN(config=args, n_direct=1)
#训练
t = Trainer(args, model, train_dataset, dev_dataset, test_dataset)
t.train()

# Save output and model
RES_PATH = 'results/'
t.save_model(t.best_ckpt, RES_PATH)