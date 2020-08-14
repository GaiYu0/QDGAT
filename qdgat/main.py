import json
import torch
import os, sys, random, numpy, pickle
import argparse
import logging
from pprint import pprint
from drop_reader import DropReader
from drop_dataloader import DropBatchGen
from network import QDGATNet
from utils import AverageMeter
from datetime import datetime
from optimizer import AdamW
from utils import create_logger
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader, RandomSampler
from drop_dataloader import create_collate_fn


logger = logging.getLogger()
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def build_network(args):
    bert_model = RobertaModel.from_pretrained(args.roberta_model)
    network = QDGATNet(bert_model,
                    hidden_size=bert_model.config.hidden_size,
                    dropout_prob=args.dropout)

    if torch.cuda.is_available():
      network.cuda()
    return network

def load(network, optimizer, path):
    network_params_path = os.path.join(path, 'checkpoint_best.pt')

    if not os.path.exists(network_params_path):
        return 0
    network.load_state_dict(torch.load(network_params_path))
    logger.info('Load network params from '+network_params_path)
    other_params_path = os.path.join(path, 'checkpoint_best.ot')
    other_params = torch.load(other_params_path)
    logger.info('Load optimizer params from '+other_params_path)
    optimizer.load_state_dict(other_params['optimizer'])
    epoch = other_params['epoch']
    best_result = other_params['best_result']
    logger.info('Load model from {} success! Current epoch is {}'.format(path, epoch))
    updates = epoch*4800
    return epoch, updates, best_result

def save(args, network, optimizer, prefix, epoch, best_result):
    network_state = dict([(k, v.cpu()) for k, v in network.state_dict().items()])
    other_params = {
        'optimizer': optimizer.state_dict(),
        'config': args,
        'epoch': epoch,
        'best_result': best_result
    }
    state_path = prefix + ".pt"
    other_path = prefix + ".ot"
    torch.save(other_params, other_path)
    torch.save(network_state, state_path)
    logger.info('model saved to {}'.format(prefix))

def preprocess_drop(args, tokenizer):
    
    for mode in ['train', 'dev']:
        cache_fpath = os.path.join(args.data_dir, "%s.pkl"%mode)
        if os.path.exists(cache_fpath): continue
        data_fpath = os.path.join(args.data_dir, "drop_dataset_number_%s_parsed.json"%mode)
        if not os.path.exists(data_fpath):
            raise Exception("Missing %s for preprocessing."%data_fpath)

        skip_when_all_empty = ["passage_span", "question_span", "addition_subtraction", "counting", "multi_span"] if mode=='train' else None
        reader = DropReader(
            tokenizer, args.passage_length_limit, args.question_length_limit,
            skip_when_all_empty=skip_when_all_empty
        )
        data = reader._read(data_fpath)
        with open(cache_fpath, "wb") as f:
            pickle.dump(data, f)
    logger.info('End of data preprocess.')


def train(args, network, train_itr, dev_itr):
    logger.info("Start training.")
    
    num_train_steps = int(args.max_epoch * len(train_itr) / args.gradient_accumulation_steps)
    logger.info("Num update steps {}!".format(num_train_steps))

    start_epoch, best_result = 1, 0.0

    metrics = {
        key:AverageMeter() for key in ['loss', 'f1', 'em']
    }
    
    def reset_metrics():
        for metric in metrics.values():
            metric.reset()

    def update_metrics(result):
        for key in metrics.keys():
            metrics[key].update(result[key])

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in network.bert.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.bert_weight_decay, 'lr': args.bert_learning_rate},
        {'params': [p for n, p in network.bert.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0, 'lr': args.bert_learning_rate},
        {'params': [p for n, p in network.named_parameters() if not n.startswith("bert.")],
            "weight_decay": args.weight_decay, "lr": args.learning_rate}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                    lr=args.learning_rate,
                    warmup=args.warmup,
                    t_total=num_train_steps,
                    max_grad_norm=args.grad_clipping,
                    schedule=args.warmup_schedule)

    update_cnt, step = 0, 0
    train_start = datetime.now()
    save_prefix = os.path.join(args.save_dir, "checkpoint_best")

    for epoch in range(start_epoch, args.max_epoch + 1):
        logger.info('Start epoch {}'.format(epoch))

        reset_metrics()
        
        for batch in train_itr:
            step += 1
            network.train()
            output_dict = network(**batch)
            loss = output_dict["loss"]
            if args.gradient_accumulation_steps > 1:
                loss /= args.gradient_accumulation_steps
            loss.backward()

            current_metrics = network.get_metrics(True)
            current_metrics['loss'] = output_dict["loss"]
            update_metrics(current_metrics)

            if (step+1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                update_cnt += 1



            if update_cnt % (args.log_per_updates * args.gradient_accumulation_steps) == 0 or update_cnt == 1:
                logger.info("QDGAT train: step:{0:6} loss:{1:.5f} f1:{2:.5f} em:{3:.5f} left:{4}".format(
                    update_cnt, metrics['loss'].avg, metrics['f1'].avg, metrics['em'].avg,
                    str((datetime.now() - train_start) / (update_cnt + 1) * (num_train_steps - update_cnt - 1)).split('.')[0]))
                reset_metrics()


        if args.do_eval:
            eval_loss, eval_f1, eval_em = evaluate(args, network, dev_itr)
            logger.info("Epoch {} eval result, loss {}, f1 {}, em {}.".format(epoch, eval_loss, eval_f1, eval_em))

        if args.do_eval and eval_f1 > best_result:
            
            save(args, network, optimizer, save_prefix, epoch, best_result)
            best_result = eval_f1
            logger.info("Best eval F1 {} at epoch {}".format(best_result, epoch))

    logger.info("Train cost {}s.".format((datetime.now() - train_start).seconds))

def evaluate(args, network, dev_itr):
    logger.info("Start evaluating.")
    
    network.eval()
    loss_metric = AverageMeter()

    eval_start = datetime.now()
    example_cnt = 0

    with torch.no_grad():
        for batch in dev_itr:
            example_cnt += batch['input_ids'].shape[0]
            output_dict = network(**batch)
            loss_metric.update(output_dict["loss"].item())
    eval_metrics = network.get_metrics(True)

    logger.info("Eval {} examples cost {}s.".format(example_cnt, (datetime.now() - eval_start).seconds))
    
    return loss_metric.avg, eval_metrics["f1"], eval_metrics["em"]

def set_environment(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--passage_length_limit", type=int, default=463)
    parser.add_argument("--question_length_limit", type=int, default=46)
    parser.add_argument("--data_dir", default="", type=str, required=True, help="The data directory.")
    parser.add_argument("--save_dir", default="", type=str, required=True, help=",The directory to save checkpoint.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--log_per_updates", default=20, type=int, help="log pre update size.")
    parser.add_argument("--do_train", action="store_true", help="Whether to do training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to do evaluation.")
    parser.add_argument("--max_epoch", default=5, type=int, help="max epoch.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="weight decay.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="learning rate.")
    parser.add_argument("--grad_clipping", default=1.0, type=float, help="gradient clip.")
    parser.add_argument('--warmup', type=float, default=0.1,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--warmup_schedule", default="warmup_linear", type=str, help="warmup schedule.")
    parser.add_argument("--optimizer", default="adam", type=str, help="train optimizer.")
    parser.add_argument('--seed', type=int, default=2018, help='random seed for data shuffling, embedding init, etc.')
    parser.add_argument('--pre_path', type=str, default=None, help="Load from pre trained.")
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout.")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size.")
    parser.add_argument('--eval_batch_size', type=int, default=32, help="eval batch size.")
    parser.add_argument("--eps", default=1e-8, type=float, help="ema gamma.")
    parser.add_argument("--bert_learning_rate", type=float, help="bert learning rate.")
    parser.add_argument("--bert_weight_decay", type=float, help="bert weight decay.")
    parser.add_argument("--roberta_model", type=str, help="robert modle path.")

    args = parser.parse_args()

    pprint(args)
    set_environment(args.seed)
    args.use_cuda = torch.cuda.is_available()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
        
    args_path = os.path.join(args.save_dir, "args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f)

    args.batch_size = args.batch_size // args.gradient_accumulation_steps

    tokenizer = RobertaTokenizer.from_pretrained(args.roberta_model)

    preprocess_drop(args, tokenizer)

    train_dataset, eval_dataset, pred_dataset = None, None, None

    if not args.do_train and not args.do_eval:
        raise Exception('Both do_train and do_eval are False.')

    collate_fn = create_collate_fn(tokenizer.pad_token_id, args.use_cuda)

    if args.do_train:
        train_dataset = DropBatchGen(args, data_mode="train", tokenizer=tokenizer)
        train_sampler = RandomSampler(train_dataset)
        train_dataset = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, num_workers=0, collate_fn=collate_fn, pin_memory=False)

    if args.do_eval:
        eval_dataset = DropBatchGen(args, data_mode="dev", tokenizer=tokenizer)
        eval_dataset = DataLoader(eval_dataset, batch_size=args.eval_batch_size, num_workers=0, collate_fn=collate_fn, pin_memory=False, shuffle=False)

    network = build_network(args)

    if args.do_train:
        train(args, network, train_dataset, eval_dataset)
    elif args.do_eval:
        evaluate(args, network, eval_dataset)

if __name__ == '__main__':
    main()
