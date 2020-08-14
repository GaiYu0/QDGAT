import os
import pickle
import torch
import random
import numpy as np
from drop_token import Token

def do_pad(batch, key, padding):
    lst = [v[key] for v in batch]
    return torch.from_numpy(np.column_stack((itertools.zip_longest(*lst, fillvalue=padding))))

def create_collate_fn(padding_idx=1, use_cuda=False):
    def collate_fn(batch):
        bsz = len(batch)
        max_seq_len = max([v['input_ids'].shape[0] for v in batch])
        input_ids = torch.LongTensor(bsz, max_seq_len).fill_(padding_idx)
        input_mask = torch.LongTensor(bsz, max_seq_len).fill_(0)
        input_segments = torch.LongTensor(bsz, max_seq_len).fill_(0)
        passage_mask = torch.LongTensor(bsz, max_seq_len).fill_(0)
        question_mask = torch.LongTensor(bsz, max_seq_len).fill_(0)

        max_num_len = max([v['number_indices'].shape[0] for v in batch])
        number_indices = torch.LongTensor(bsz, max_num_len).fill_(-1)

        max_qnum_len = max([v['question_number_indices'].shape[0] for v in batch])
        question_number_indices = torch.LongTensor(bsz, max_qnum_len).fill_(-1)
        passage_number_order = torch.LongTensor(bsz, max_num_len).fill_(-1)
        question_number_order = torch.LongTensor(bsz, max_qnum_len).fill_(-1)


        max_gnodes_choice = max([v['gnodes'].shape[0] for v in batch])
        max_gnodes_len_choice = max([v['gnodes_len'].shape[0] for v in batch])
        max_gnodes_mask_choice = max([v['gnodes_mask'].shape[0] for v in batch])
        max_gnodes_mask_len_choice = max([v['gnodes_mask'].shape[1] for v in batch])
        max_gedges_choice = max([v['gedges'].shape[0] for v in batch])

        gnodes = torch.LongTensor(bsz, max_gnodes_choice).fill_(-1)
        gnodes_len = torch.LongTensor(bsz, max_gnodes_len_choice).fill_(-1)
        gnodes_mask = torch.LongTensor(bsz, max_gnodes_mask_choice, max_gnodes_mask_len_choice).fill_(-1)
        gedges = torch.LongTensor(bsz, max_gnodes_len_choice, max_gnodes_len_choice).fill_(0)

        max_pans_choice = max([v['answer_as_passage_spans'].shape[0] for v in batch])
        answer_as_passage_spans = torch.LongTensor(bsz, max_pans_choice, 2).fill_(-1)

        max_qans_choice = max([v['answer_as_question_spans'].shape[0] for v in batch])
        answer_as_question_spans = torch.LongTensor(bsz, max_qans_choice, 2).fill_(-1)

        max_sign_choice = max([v['answer_as_add_sub_expressions'].shape[0] for v in batch])
        answer_as_add_sub_expressions = torch.LongTensor(bsz, max_sign_choice, max_num_len).fill_(0)
        answer_as_counts = torch.LongTensor(bsz).fill_(-1)
        max_text_answers = max([v['answer_as_text_to_disjoint_bios'].shape[0] for v in batch])
        max_answer_spans = max([v['answer_as_text_to_disjoint_bios'].shape[1] for v in batch])
        answer_as_text_to_disjoint_bios = torch.LongTensor(bsz, max_text_answers, max_answer_spans, max_seq_len).fill_(0)
        max_correct_sequences = max([v['answer_as_list_of_bios'].shape[0] for v in batch])
        answer_as_list_of_bios = torch.LongTensor(bsz, max_correct_sequences, max_seq_len).fill_(0)
        span_bio_labels = torch.LongTensor(bsz, max_seq_len).fill_(0)
        bio_wordpiece_mask = torch.LongTensor(bsz, max_seq_len).fill_(0)
        is_bio_mask = torch.LongTensor(bsz).fill_(0)

        batch_gedge_types = 0
        for i in range(bsz):
            item = batch[i]
            try:
                seq_len = item['input_ids'].shape[0]
                input_ids[i][:seq_len] = torch.LongTensor(item['input_ids'])
                input_mask[i][:seq_len] = torch.LongTensor(item['input_mask'])
                input_segments[i][:seq_len] = torch.LongTensor(item['input_segments'])
                pm_len = item["passage_mask"].shape[0]
                passage_mask[i][:pm_len] = torch.LongTensor(item["passage_mask"])
                qm_len = item["question_mask"].shape[0]
                question_mask[i][:qm_len] = torch.LongTensor(item["question_mask"])

                number_indices[i][:item["number_indices"].shape[0]] = torch.LongTensor(item["number_indices"])
                question_number_indices[i][:item["question_number_indices"].shape[0]] = torch.LongTensor(item["question_number_indices"])
                passage_number_order[i][:item["passage_number_order"].shape[0]] = torch.LongTensor(item["passage_number_order"])
                question_number_order[i][:item["question_number_order"].shape[0]] = torch.LongTensor(item["question_number_order"])

                gn_len = item["gnodes"].shape[0]
                gnodes[i][:gn_len] = torch.LongTensor(item["gnodes"])
                gnodes_len[i][:item["gnodes_len"].shape[0]] = torch.LongTensor(item["gnodes_len"])
                gnodes_mask[i][:gn_len, :item["gnodes_mask"].shape[1]] = torch.LongTensor(item["gnodes_mask"])
                gedges[i][:gn_len, :gn_len] = torch.LongTensor(item["gedges"])
                batch_gedge_types |= item['gedge_types']


                answer_as_passage_spans[i][:item["answer_as_passage_spans"].shape[0],:] = torch.LongTensor(item["answer_as_passage_spans"])
                answer_as_question_spans[i][:item["answer_as_question_spans"].shape[0],:] = torch.LongTensor(item["answer_as_question_spans"])

                add_sub_len, num_len = item['answer_as_add_sub_expressions'].shape
                answer_as_add_sub_expressions[i][:add_sub_len,:num_len] = torch.LongTensor(item['answer_as_add_sub_expressions'])

                answer_as_counts[i] = torch.LongTensor(item['answer_as_counts'])

                s0,s1,s2 = item['answer_as_text_to_disjoint_bios'].shape
                answer_as_text_to_disjoint_bios[i][:s0, :s1, :s2] = torch.LongTensor(item['answer_as_text_to_disjoint_bios'])

                s0, s1 = item['answer_as_list_of_bios'].shape
                answer_as_list_of_bios[i][:s0, :s1] = torch.LongTensor(item['answer_as_list_of_bios'])
                span_bio_labels[i][:item['span_bio_labels'].shape[0]] = torch.LongTensor(item['span_bio_labels'])

                is_bio_mask[i] = torch.LongTensor(item['is_bio_mask'])
                bio_wordpiece_mask[i][:item['bio_wordpiece_mask'].shape[0]] = torch.LongTensor(item['bio_wordpiece_mask'])
            except Exception as e:
                import ipdb
                ipdb.set_trace()
                x=1

        out_batch = {"input_ids": input_ids, "input_mask": input_mask, "input_segments": input_segments,
               "passage_mask": passage_mask, "question_mask": question_mask, "number_indices": number_indices,
               "passage_number_order": passage_number_order,
               "question_number_order": question_number_order,
               "question_number_indices": question_number_indices,
               "gnodes": gnodes,
               "gnodes_len": gnodes_len,
               "gnodes_mask": gnodes_mask,
               "gedges": gedges,
               "gedge_types": batch_gedge_types,
               "answer_as_passage_spans": answer_as_passage_spans,
               "answer_as_question_spans": answer_as_question_spans,
               "answer_as_add_sub_expressions": answer_as_add_sub_expressions,
               "answer_as_counts": answer_as_counts.unsqueeze(1),
               "answer_as_text_to_disjoint_bios": answer_as_text_to_disjoint_bios,
               "answer_as_list_of_bios": answer_as_list_of_bios,
               "span_bio_labels": span_bio_labels,
               "is_bio_mask": is_bio_mask,
               "bio_wordpiece_mask": bio_wordpiece_mask,
               "metadata": [v['metadata'] for v in batch]}

        if use_cuda:
            for k in out_batch.keys():
                if isinstance(out_batch[k], torch.Tensor):
                    out_batch[k] = out_batch[k].cuda()
        return out_batch


    return collate_fn

class DropDataBuilder(object):
    def __init__(self, args, data_mode, tokenizer, padding_idx=1):
        self.args = args
        self.cls_idx = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        self.sep_idx = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        self.padding_idx = padding_idx
        self.is_train = data_mode == "train"
        self.vocab_size = len(tokenizer)
        dpath = "{}.pkl".format(data_mode)
        with open(os.path.join(args.data_dir, dpath), "rb") as f:
            print("Load data from {}.".format(dpath))
            data = pickle.load(f)

        raw_data = []
        for item in data:
            question_tokens = tokenizer.convert_tokens_to_ids(item["question_tokens"])
            passage_tokens = tokenizer.convert_tokens_to_ids(item["passage_tokens"])
            question_passage_tokens = [ Token(text=item[0], idx=item[1][0], edx=item[1][1] ) for item in zip(item["question_passage_tokens"],
                    [(0,0)] + item["question_token_offsets"] + [(0,0)]+ item["passage_token_offsets"] + [(0, 0)])]
            item["question_passage_tokens"] = question_passage_tokens
            raw_data.append((question_tokens, passage_tokens, item))

        self.data = [self.build(item) for item in raw_data]
        del raw_data

        print("Load data size {}.".format(len(self.data)))

        self.offset = 0



    def __len__(self):
        return len(self.data)

    def get(self, offset):
        return self.data[offset]

    def build(self, batch):
        q_tokens, p_tokens, metas = batch
        seq_len = len(q_tokens) + len(p_tokens) + 3
        num_len = max(1, len(metas["number_indices"]))
        qnum_len = max(1, len(metas["question_number_indices"]))

        pans_choice = max(1, len(metas["answer_passage_spans"]))
        qans_choice = max(1, len(metas["answer_question_spans"]))
        sign_choice = max(1, len(metas["signs_for_add_sub_expressions"]))

        # qa input.
        input_segments = np.zeros(seq_len)

        # multiple span label
        max_text_answers = max(1, 0 if len(metas["multi_span"]) < 1 else
                                   len(metas["multi_span"][1]))
        max_answer_spans = max(1, 0 if len(metas["multi_span"]) < 1 else
                                   max([len(item) for item in metas["multi_span"][1]])
                                   )
        max_correct_sequences = max(1, 0 if len(metas["multi_span"]) < 1 else
                                   len(metas["multi_span"][2])
                                   )
        bio_wordpiece_mask = np.zeros([seq_len], dtype=np.int)
        answer_as_text_to_disjoint_bios = np.zeros([max_text_answers, max_answer_spans, seq_len])
        answer_as_list_of_bios = np.zeros([max_correct_sequences, seq_len])
        span_bio_labels = np.zeros([seq_len])

        answer_as_passage_spans = np.full([pans_choice, 2], -1)
        answer_as_question_spans = np.full([qans_choice, 2], -1)

        answer_as_add_sub_expressions = np.zeros([sign_choice, num_len], dtype=np.int)

        q_len = len(q_tokens)
        p_len = len(p_tokens)
        # input and their mask
        input_ids = np.array([self.cls_idx] + q_tokens + [self.sep_idx] + p_tokens + [self.sep_idx])
        input_mask = np.ones(3 + q_len + p_len, dtype=np.int)
        question_mask = np.zeros(seq_len)
        question_mask[1:1 + q_len] = 1
        passage_mask = np.zeros(seq_len)

        passage_start = q_len + 2
        passage_mask[passage_start: passage_start + p_len] = 1


        gnode_cnt = max(1, len(metas["gnodes"]))
        gnodes = np.full([gnode_cnt], -1)
        gnodes_len = np.full([gnode_cnt], -1)
        gnodes_mask_len = max(1, len(metas["gnodes_mask"][0]))
        gnodes_mask = np.full([gnode_cnt, gnodes_mask_len], -1)
        gedges = np.full([gnode_cnt, gnode_cnt], 0)

        question_start = 1
        # number infos
        pn_len = len(metas["number_indices"]) - 1
        number_indices = np.full([num_len], -1)
        passage_number_order = np.full([num_len], -1)
        if pn_len > 0:
            number_indices[:pn_len] = passage_start + np.array(metas["number_indices"][:pn_len])
            passage_number_order[:pn_len] = np.array(metas["passage_number_order"][:pn_len])
            number_indices[pn_len - 1] = 0
        qn_len = len(metas["question_number_indices"]) - 1
        question_number_order = np.full([qnum_len], -1)
        question_number_indices = np.full([qnum_len], -1)
        if qn_len > 0:
            question_number_indices[:qn_len] = question_start + np.array(
                metas["question_number_indices"][:qn_len])
            question_number_order[:qn_len] = np.array(metas["question_number_order"][:qn_len])


        gn_len = len(metas["gnodes"]) - 1
        if gn_len > 0:
            gnodes[:gn_len] = passage_start + np.array(metas["gnodes"][:gn_len])
            gnodes_len[:gn_len] = np.array(metas["gnodes_len"][:gn_len])
            mask_dim = len(metas['gnodes_mask'][0])
            gnodes_mask_raw = np.array(metas['gnodes_mask'][:gn_len])
            mask = (gnodes_mask_raw >= 0).astype(int)
            gnodes_mask[:gn_len, :mask_dim] = (passage_start + gnodes_mask_raw)*mask

        batch_gedge_types = metas['gedge_types']
        ge_len = len(metas["gedges"][0]) - 1
        if ge_len > 0:
            z0,z1,z2=np.array(metas["gedges"])[:,:ge_len]
            gedges[z0, z1]=z2
            gedges[z1, z0]=z2



        # answer info
        pans_len = min(len(metas["answer_passage_spans"]), pans_choice)
        for j in range(pans_len):
            answer_as_passage_spans[j, 0] = np.array(metas["answer_passage_spans"][j][0]) + passage_start
            answer_as_passage_spans[j, 1] = np.array(metas["answer_passage_spans"][j][1]) + passage_start

        qans_len = min(len(metas["answer_question_spans"]), qans_choice)
        for j in range(qans_len):
            answer_as_question_spans[j, 0] = np.array(metas["answer_question_spans"][j][0]) + question_start
            answer_as_question_spans[j, 1] = np.array(metas["answer_question_spans"][j][1]) + question_start

        # answer sign info
        sign_len = min(len(metas["signs_for_add_sub_expressions"]), sign_choice)
        for j in range(sign_len):
            answer_as_add_sub_expressions[j, :pn_len] = np.array(
                metas["signs_for_add_sub_expressions"][j][:pn_len])

        # answer count info
        if len(metas["counts"]) > 0:
            answer_as_counts = np.array(metas["counts"])
        else:
            answer_as_counts = np.array([-1])

        # add multi span prediction
        cur_seq_len = q_len + p_len + 3
        bio_wordpiece_mask[:cur_seq_len] = np.array(metas["wordpiece_mask"][:cur_seq_len])
        is_bio_mask = np.zeros([1])
        if len(metas["multi_span"]) > 0:
            is_bio_mask[0] = metas["multi_span"][0]
            span_bio_labels[:cur_seq_len] = np.array(metas["multi_span"][-1][:cur_seq_len])
            for j in range(len(metas["multi_span"][1])):
                for k in range(len(metas["multi_span"][1][j])):
                    answer_as_text_to_disjoint_bios[j, k, :cur_seq_len] = np.array(metas["multi_span"][1][j][k][:cur_seq_len])
            for j in range(len(metas["multi_span"][2])):
                answer_as_list_of_bios[j, :cur_seq_len] = np.array(metas["multi_span"][2][j][:cur_seq_len])

        out_batch = {"input_ids": input_ids, "input_mask": input_mask, "input_segments": input_segments,
                     "passage_mask": passage_mask, "question_mask": question_mask, "number_indices": number_indices,
                     "passage_number_order": passage_number_order,
                     "question_number_order": question_number_order,
                     "question_number_indices": question_number_indices,
                     "gnodes": gnodes,
                     "gnodes_len": gnodes_len,
                     "gnodes_mask": gnodes_mask,
                     "gedges": gedges,
                     "gedge_types": batch_gedge_types,
                     "answer_as_passage_spans": answer_as_passage_spans,
                     "answer_as_question_spans": answer_as_question_spans,
                     "answer_as_add_sub_expressions": answer_as_add_sub_expressions,
                     "answer_as_counts": np.expand_dims(answer_as_counts, axis=1),
                     "answer_as_text_to_disjoint_bios": answer_as_text_to_disjoint_bios,
                     "answer_as_list_of_bios": answer_as_list_of_bios,
                     "span_bio_labels": span_bio_labels,
                     "is_bio_mask": is_bio_mask,
                     "bio_wordpiece_mask": bio_wordpiece_mask,
                     "metadata": metas}

        return out_batch

class DropBatchGen(object):
    def __init__(self, args, data_mode, tokenizer, padding_idx=1):
        self.gen = DropDataBuilder(args, data_mode, tokenizer, padding_idx)

    def __getitem__(self, index):
        return self.gen.get(index)

    def __len__(self):
        return len(self.gen.data)

    def reset(self):
        return self.gen.reset()
