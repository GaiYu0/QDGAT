import torch, datetime
import torch.nn as nn
from typing import List, Dict, Any
from collections import OrderedDict
import torch.nn.functional as F

import allennlp_util as util
from utils import DropEmAndF1, format_number
from qdgat import ResidualGRU, BertFeedForward, QDGAT

from multispan_heads import multispan_heads_mapping, remove_substring_from_prediction

class QDGATNet(nn.Module):
    """
    This class augments the QANet model with some rudimentary numerical reasoning abilities, as
    published in the original DROP paper.

    The main idea here is that instead of just predicting a passage span after doing all of the
    QANet modeling stuff, we add several different "answer abilities": predicting a span from the
    question, predicting a count, or predicting an arithmetic expression.  Near the end of the
    QANet model, we have a variable that predicts what kind of answer type we need, and each branch
    has separate modeling logic to predict that answer type.  We then marginalize over all possible
    ways of getting to the right answer through each of these answer types.
    """
    def __init__(self,
                 bert,
                 hidden_size: int,
                 dropout_prob: float = 0.1,
                 answering_abilities: List[str] = None,
                 use_gcn: bool = False,
                 gcn_steps: int = 1,
                 unique_on_multispan: bool = True,
                 multispan_head_name: str = "flexible_loss",
                 multispan_generation_top_k: int = 0,
                 multispan_prediction_beam_size: int = 5,
                 multispan_use_prediction_beam_search: bool = False,
                 multispan_use_bio_wordpiece_mask: bool = True,
                 dont_add_substrings_to_ms: bool = True,
                 trainning: bool = True,
                 ) -> None:
        super(QDGATNet, self).__init__()
        self.training = trainning
        self.multispan_head_name = multispan_head_name
        self.multispan_use_prediction_beam_search = multispan_use_prediction_beam_search
        self.multispan_use_bio_wordpiece_mask = multispan_use_bio_wordpiece_mask
        self._dont_add_substrings_to_ms = dont_add_substrings_to_ms

        self.use_gcn = use_gcn
        self.bert = bert
        modeling_out_dim = hidden_size
        self._drop_metrics = DropEmAndF1()
        if answering_abilities is None:
            self.answering_abilities = ["passage_span_extraction", "question_span_extraction",
                                        "addition_subtraction", "counting", "multiple_spans"]
        else:
            self.answering_abilities = answering_abilities

        if len(self.answering_abilities) > 1:
            self._answer_ability_predictor = BertFeedForward(3 * hidden_size, hidden_size, len(self.answering_abilities))

        if "passage_span_extraction" in self.answering_abilities or "question_span_extraction" in self.answering_abilities:
            self._passage_span_extraction_index = self.answering_abilities.index("passage_span_extraction")
            self._question_span_extraction_index = self.answering_abilities.index("question_span_extraction")
            self._span_start_predictor = nn.Linear ( 4 * hidden_size, 1, bias=False)
            self._span_end_predictor = nn.Linear(4 * hidden_size, 1, bias=False)

        if "addition_subtraction" in self.answering_abilities:
            self._addition_subtraction_index = self.answering_abilities.index("addition_subtraction")
            self._number_sign_predictor = BertFeedForward(5 * hidden_size, hidden_size, 3)

        if "counting" in self.answering_abilities:
            self._counting_index = self.answering_abilities.index("counting")
            self._count_number_predictor = BertFeedForward(5 * hidden_size, hidden_size, 10)

        # add multiple span prediction from https://github.com/eladsegal/project-NLP-AML
        if "multiple_spans" in self.answering_abilities:
            if self.multispan_head_name == "flexible_loss":
                self.multispan_head = multispan_heads_mapping[multispan_head_name](modeling_out_dim,
                                                                                   generation_top_k=multispan_generation_top_k,
                                                                                   prediction_beam_size=multispan_prediction_beam_size)
            else:
                self.multispan_head = multispan_heads_mapping[multispan_head_name](modeling_out_dim)

            self._multispan_module = self.multispan_head.module
            self._multispan_log_likelihood = self.multispan_head.log_likelihood
            self._multispan_prediction = self.multispan_head.prediction
            self._unique_on_multispan = unique_on_multispan

        self._dropout = torch.nn.Dropout(p=dropout_prob)

        if self.use_gcn:
            node_dim = modeling_out_dim

            self._gcn_input_proj = nn.Linear(node_dim * 2, node_dim)
            self._iteration_steps = gcn_steps
            print('gcn iteration_steps=%d' % self._iteration_steps, flush=True)
            self._proj_ln = nn.LayerNorm(node_dim)
            self._proj_ln0 = nn.LayerNorm(node_dim)
            self._proj_ln1 = nn.LayerNorm(node_dim)
            self._proj_ln3 = nn.LayerNorm(node_dim)

            self._gcn_enc = ResidualGRU(hidden_size, dropout_prob, 2)
        # add bert proj
        self._proj_sequence_h = nn.Linear(hidden_size, 1, bias=False)
        self._proj_number = nn.Linear(hidden_size*2, 1, bias=False)
        self._proj_date = nn.Linear(hidden_size*2, 1, bias=False)

        self._proj_sequence_g0 = BertFeedForward(hidden_size, hidden_size, 1)
        self._proj_sequence_g1 = BertFeedForward(hidden_size, hidden_size, 1)
        self._proj_sequence_g2 = BertFeedForward(hidden_size, hidden_size, 1)

        #GAT
#        self.gat = GAT(node_dim, node_dim, dropout=0.6, alpha=0.2, nheads=2, edge_type_num=9)
        node_dim = modeling_out_dim
        self.qdgat = QDGAT(node_dim, edge_type_num=9, iter_num=4)

    def forward(self,  # type: ignore
                input_ids: torch.LongTensor,
                input_mask: torch.LongTensor,
                input_segments: torch.LongTensor,
                passage_mask: torch.LongTensor,
                question_mask: torch.LongTensor,
                number_indices: torch.LongTensor,
                passage_number_order: torch.LongTensor,
                question_number_order: torch.LongTensor,
                question_number_indices: torch.LongTensor,
                gnodes: torch.LongTensor,
                gnodes_len: torch.LongTensor,
                gnodes_mask: torch.LongTensor,
                gedges: torch.LongTensor,
                gedge_types: int,
                answer_as_passage_spans: torch.LongTensor = None,
                answer_as_question_spans: torch.LongTensor = None,
                answer_as_add_sub_expressions: torch.LongTensor = None,
                answer_as_counts: torch.LongTensor = None,
                answer_as_text_to_disjoint_bios: torch.LongTensor = None,
                answer_as_list_of_bios: torch.LongTensor = None,
                span_bio_labels: torch.LongTensor = None,
                bio_wordpiece_mask: torch.LongTensor = None,
                is_bio_mask: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:

        outputs = self.bert(input_ids, attention_mask=input_mask, token_type_ids=input_segments)
        sequence_output = outputs[0]
        sequence_output_list = [ item for item in outputs[2][-4:] ]

        batch_size = input_ids.size(0)
        if ("passage_span_extraction" in self.answering_abilities or "question_span" in self.answering_abilities) and self.use_gcn:
            # M2, M3
            sequence_alg = self._gcn_input_proj(torch.cat([sequence_output_list[2], sequence_output_list[3]], dim=2))
            encoded_passage_for_numbers = sequence_alg
            encoded_question_for_numbers = sequence_alg
            encoded_dates = sequence_alg
            encoded_passage = sequence_alg

            #gen embedding
            real_gnodes_indices = gnodes_mask - 1
            real_gnodes_mask = gnodes_mask>0
            real_gnodes_indices_col_len = real_gnodes_indices.size(-1)
            z0 = real_gnodes_indices.view(batch_size, -1)
            #gnodes will large than 0, since passage_start exist
            z1=(z0>-9999).nonzero()[:,0].view(batch_size,-1)
            gnodes_embs = encoded_passage[z1,z0]
            gnodes_embs = gnodes_embs.view(batch_size,-1,real_gnodes_indices_col_len,encoded_passage.size(-1))
            gnodes_token_ids = real_gnodes_indices

            real_gnodes_mask = real_gnodes_mask.unsqueeze(-1).expand(-1,-1, -1, encoded_passage.size(-1))
            gnodes_embs = util.replace_masked_values(gnodes_embs, real_gnodes_mask, 0)
            sum_gnodes_embs = gnodes_embs.sum(-2)
            mean_gnodes_embs = sum_gnodes_embs/gnodes_len.unsqueeze(-1).float()
            gnodes_mask_bin = (gnodes_len > 0).long()
            cls_emb = sequence_output[:, 0]
            new_gnodes_embs = self.qdgat(mean_gnodes_embs, gnodes_mask_bin, cls_emb, gedges, gedge_types, is_print=False)


            gnodes_embs_updated = util.replace_masked_values(new_gnodes_embs.unsqueeze(-2).expand(-1,-1,real_gnodes_indices_col_len,-1), real_gnodes_mask, 0)
            gnodes_embs_updated = gnodes_embs_updated.view(batch_size, -1, sequence_alg.size(-1))

            gcn_info_vec = torch.zeros(encoded_passage.shape, dtype=torch.float, device=gnodes.device)

            gnodes_updated_index = util.replace_masked_values(z0, z0>0, 0)
            gcn_info_vec.scatter_(1, gnodes_updated_index.unsqueeze(-1).expand(-1, -1, gnodes_embs_updated.size(-1)), gnodes_embs_updated)

            sequence_output_list[2] = self._gcn_enc(self._proj_ln(sequence_output_list[2] + gcn_info_vec))
            sequence_output_list[0] = self._gcn_enc(self._proj_ln0(sequence_output_list[0] + gcn_info_vec))
            sequence_output_list[1] = self._gcn_enc(self._proj_ln1(sequence_output_list[1] + gcn_info_vec))
            sequence_output_list[3] = self._gcn_enc(self._proj_ln3(sequence_output_list[3] + gcn_info_vec))


        # passage hidden and question hidden
        sequence_h2_weight = self._proj_sequence_h(sequence_output_list[2]).squeeze(-1)
        passage_h2_weight = util.masked_softmax(sequence_h2_weight, passage_mask)
        passage_h2 = util.weighted_sum(sequence_output_list[2], passage_h2_weight)
        question_h2_weight = util.masked_softmax(sequence_h2_weight, question_mask)
        question_h2 = util.weighted_sum(sequence_output_list[2], question_h2_weight)

        # passage g0, g1, g2
        question_g0_weight = self._proj_sequence_g0(sequence_output_list[0]).squeeze(-1)
        question_g0_weight = util.masked_softmax(question_g0_weight, question_mask)
        question_g0 = util.weighted_sum(sequence_output_list[0], question_g0_weight)

        question_g1_weight = self._proj_sequence_g1(sequence_output_list[1]).squeeze(-1)
        question_g1_weight = util.masked_softmax(question_g1_weight, question_mask)
        question_g1 = util.weighted_sum(sequence_output_list[1], question_g1_weight)

        question_g2_weight = self._proj_sequence_g2(sequence_output_list[2]).squeeze(-1)
        question_g2_weight = util.masked_softmax(question_g2_weight, question_mask)
        question_g2 = util.weighted_sum(sequence_output_list[2], question_g2_weight)


        if len(self.answering_abilities) > 1:
            # Shape: (batch_size, number_of_abilities)
            answer_ability_logits = self._answer_ability_predictor(torch.cat([passage_h2, question_h2, sequence_output[:, 0]], 1))
            answer_ability_log_probs = F.log_softmax(answer_ability_logits, -1)
            best_answer_ability = torch.argmax(answer_ability_log_probs, 1)
            top_two_answer_abilities = torch.topk(answer_ability_log_probs, k=2, dim=1)

        real_number_indices = number_indices.squeeze(-1) - 1
        number_mask = (real_number_indices > -1).long()
        clamped_number_indices = util.replace_masked_values(real_number_indices, number_mask, 0)
        encoded_passage_for_numbers = torch.cat([sequence_output_list[2], sequence_output_list[3]], dim=-1)

        encoded_numbers = torch.gather(encoded_passage_for_numbers, 1,
            clamped_number_indices.unsqueeze(-1).expand(-1, -1, encoded_passage_for_numbers.size(-1)))
        number_weight = self._proj_number(encoded_numbers).squeeze(-1)
        number_mask = (number_indices > -1).long()
        number_weight = util.masked_softmax(number_weight, number_mask)
        number_vector = util.weighted_sum(encoded_numbers, number_weight)



        if "counting" in self.answering_abilities:
            # Shape: (batch_size, 10)
            count_number_logits = self._count_number_predictor(torch.cat([number_vector, passage_h2, question_h2, sequence_output[:, 0]], dim=1))
            count_number_log_probs = torch.nn.functional.log_softmax(count_number_logits, -1)
            # Info about the best count number prediction
            # Shape: (batch_size,)
            best_count_number = torch.argmax(count_number_log_probs, -1)
            best_count_log_prob = torch.gather(count_number_log_probs, 1, best_count_number.unsqueeze(-1)).squeeze(-1)
            if len(self.answering_abilities) > 1:
                best_count_log_prob += answer_ability_log_probs[:, self._counting_index]

        if "passage_span_extraction" in self.answering_abilities or "question_span_extraction" in self.answering_abilities:
            # start 0, 2
            sequence_for_span_start = torch.cat([sequence_output_list[2],
                                                 sequence_output_list[0],
                                                 sequence_output_list[2]*question_g2.unsqueeze(1),
                                                 sequence_output_list[0]*question_g0.unsqueeze(1)],
                                     dim=2)
            sequence_span_start_logits = self._span_start_predictor(sequence_for_span_start).squeeze(-1)
            # Shape: (batch_size, passage_length, modeling_dim * 2)
            sequence_for_span_end = torch.cat([sequence_output_list[2],
                                               sequence_output_list[1],
                                               sequence_output_list[2]*question_g2.unsqueeze(1),
                                               sequence_output_list[1]*question_g1.unsqueeze(1)],
                                            dim=2)
            # Shape: (batch_size, passage_length)
            sequence_span_end_logits = self._span_end_predictor(sequence_for_span_end).squeeze(-1)
            # Shape: (batch_size, passage_length)

            if "passage_span_extraction" in self.answering_abilities:
                passage_span_start_log_probs = util.masked_log_softmax(sequence_span_start_logits, passage_mask)
                passage_span_end_log_probs = util.masked_log_softmax(sequence_span_end_logits, passage_mask)

                # Info about the best passage span prediction
                passage_span_start_logits = util.replace_masked_values(sequence_span_start_logits, passage_mask, -1e7)
                passage_span_end_logits = util.replace_masked_values(sequence_span_end_logits, passage_mask, -1e7)
                # Shage: (batch_size, topk, 2)
                best_passage_span = util.get_best_span(passage_span_start_logits, passage_span_end_logits)

            if "question_span_extraction" in self.answering_abilities:
                question_span_start_log_probs = util.masked_log_softmax(sequence_span_start_logits, question_mask)
                question_span_end_log_probs = util.masked_log_softmax(sequence_span_end_logits, question_mask)

                # Info about the best question span prediction
                question_span_start_logits = util.replace_masked_values(sequence_span_start_logits, question_mask, -1e7)
                question_span_end_logits = util.replace_masked_values(sequence_span_end_logits, question_mask, -1e7)
                # Shape: (batch_size, topk, 2)
                best_question_span = util.get_best_span(question_span_start_logits, question_span_end_logits)


        if "addition_subtraction" in self.answering_abilities:
            alg_encoded_numbers = torch.cat(
                [encoded_numbers,
                 question_h2.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1),
                 passage_h2.unsqueeze(1).repeat(1, encoded_numbers.size(1), 1),
                 sequence_output[:, 0].unsqueeze(1).repeat(1, encoded_numbers.size(1), 1)
                 ], 2)

            # Shape: (batch_size, # of numbers in the passage, 3)
            number_sign_logits = self._number_sign_predictor(alg_encoded_numbers)
            number_sign_log_probs = torch.nn.functional.log_softmax(number_sign_logits, -1)

            # Shape: (batch_size, # of numbers in passage).
            best_signs_for_numbers = torch.argmax(number_sign_log_probs, -1)
            # For padding numbers, the best sign masked as 0 (not included).
            best_signs_for_numbers = util.replace_masked_values(best_signs_for_numbers, number_mask, 0)
            # Shape: (batch_size, # of numbers in passage)
            best_signs_log_probs = torch.gather(number_sign_log_probs, 2, best_signs_for_numbers.unsqueeze(-1)).squeeze(
                -1)
            # the probs of the masked positions should be 1 so that it will not affect the joint probability
            # TODO: this is not quite right, since if there are many numbers in the passage,
            # TODO: the joint probability would be very small.
            best_signs_log_probs = util.replace_masked_values(best_signs_log_probs, number_mask, 0)
            # Shape: (batch_size,)
            best_combination_log_prob = best_signs_log_probs.sum(-1)
            if len(self.answering_abilities) > 1:
                best_combination_log_prob += answer_ability_log_probs[:, self._addition_subtraction_index]

        # add multiple span prediction
        if bio_wordpiece_mask is None or not self.multispan_use_bio_wordpiece_mask:
            multispan_mask = input_mask
        else:
            multispan_mask = input_mask * bio_wordpiece_mask
        if "multiple_spans" in self.answering_abilities:
            if self.multispan_head_name == "flexible_loss":
                multispan_log_probs, multispan_logits = self._multispan_module(sequence_output, seq_mask=multispan_mask)
            else:
                multispan_log_probs, multispan_logits = self._multispan_module(sequence_output)

        output_dict = {}

        # If answer is given, compute the loss.
        if answer_as_passage_spans is not None or answer_as_question_spans is not None or answer_as_add_sub_expressions is not None or answer_as_counts is not None:

            log_marginal_likelihood_list = []

            for answering_ability in self.answering_abilities:
                if answering_ability == "passage_span_extraction":
                    # Shape: (batch_size, # of answer spans)
                    gold_passage_span_starts = answer_as_passage_spans[:, :, 0]
                    gold_passage_span_ends = answer_as_passage_spans[:, :, 1]
                    # Some spans are padded with index -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    gold_passage_span_mask = (gold_passage_span_starts != -1).long()
                    clamped_gold_passage_span_starts = util.replace_masked_values(gold_passage_span_starts,
                                                                                  gold_passage_span_mask, 0)
                    clamped_gold_passage_span_ends = util.replace_masked_values(gold_passage_span_ends,
                                                                                gold_passage_span_mask, 0)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_passage_span_starts = torch.gather(passage_span_start_log_probs, 1,
                                                                          clamped_gold_passage_span_starts)
                    log_likelihood_for_passage_span_ends = torch.gather(passage_span_end_log_probs, 1,
                                                                        clamped_gold_passage_span_ends)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_passage_spans = log_likelihood_for_passage_span_starts + log_likelihood_for_passage_span_ends
                    # For those padded spans, we set their log probabilities to be very small negative value
                    log_likelihood_for_passage_spans = util.replace_masked_values(log_likelihood_for_passage_spans,
                                                                                  gold_passage_span_mask, -1e7)
                    # Shape: (batch_size, )
                    log_marginal_likelihood_for_passage_span = util.logsumexp(log_likelihood_for_passage_spans)

#                    print('passage_span_extraction: ', log_marginal_likelihood_for_passage_span)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_passage_span)

                elif answering_ability == "question_span_extraction":
                    # Shape: (batch_size, # of answer spans)
                    gold_question_span_starts = answer_as_question_spans[:, :, 0]
                    gold_question_span_ends = answer_as_question_spans[:, :, 1]
                    # Some spans are padded with index -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    gold_question_span_mask = (gold_question_span_starts != -1).long()
                    clamped_gold_question_span_starts = util.replace_masked_values(gold_question_span_starts,
                                                                                   gold_question_span_mask, 0)
                    clamped_gold_question_span_ends = util.replace_masked_values(gold_question_span_ends,
                                                                                 gold_question_span_mask, 0)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_question_span_starts = torch.gather(question_span_start_log_probs, 1,
                                                                           clamped_gold_question_span_starts)
                    log_likelihood_for_question_span_ends = torch.gather(question_span_end_log_probs, 1,
                                                                         clamped_gold_question_span_ends)
                    # Shape: (batch_size, # of answer spans)
                    log_likelihood_for_question_spans = log_likelihood_for_question_span_starts + log_likelihood_for_question_span_ends
                    # For those padded spans, we set their log probabilities to be very small negative value
                    log_likelihood_for_question_spans = util.replace_masked_values(log_likelihood_for_question_spans,
                                                                                   gold_question_span_mask, -1e7)
                    # Shape: (batch_size, )
                    # pylint: disable=invalid-name
                    log_marginal_likelihood_for_question_span = util.logsumexp(log_likelihood_for_question_spans)

                    # question multi span prediction
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_question_span)
#                    print('log_marginal_likelihood_for_question_span: ', log_marginal_likelihood_for_question_span)

                elif answering_ability == "addition_subtraction":
                    # The padded add-sub combinations use 0 as the signs for all numbers, and we mask them here.
                    # Shape: (batch_size, # of combinations)
                    gold_add_sub_mask = (answer_as_add_sub_expressions.sum(-1) > 0).float()
                    # Shape: (batch_size, # of numbers in the passage, # of combinations)
                    gold_add_sub_signs = answer_as_add_sub_expressions.transpose(1, 2)
                    # Shape: (batch_size, # of numbers in the passage, # of combinations)
                    log_likelihood_for_number_signs = torch.gather(number_sign_log_probs, 2, gold_add_sub_signs)
                    # the log likelihood of the masked positions should be 0
                    # so that it will not affect the joint probability
                    log_likelihood_for_number_signs = util.replace_masked_values(log_likelihood_for_number_signs,
                                                                                 number_mask.unsqueeze(-1), 0)
                    # Shape: (batch_size, # of combinations)
                    log_likelihood_for_add_subs = log_likelihood_for_number_signs.sum(1)
                    # For those padded combinations, we set their log probabilities to be very small negative value
                    log_likelihood_for_add_subs = util.replace_masked_values(log_likelihood_for_add_subs,
                                                                             gold_add_sub_mask, -1e7)
                    # Shape: (batch_size, )
                    log_marginal_likelihood_for_add_sub = util.logsumexp(log_likelihood_for_add_subs)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_add_sub)
#                    print('log_marginal_likelihood_for_add_sub: ', log_marginal_likelihood_for_add_sub)

                elif answering_ability == "counting":
                    # Count answers are padded with label -1,
                    # so we clamp those paddings to 0 and then mask after `torch.gather()`.
                    # Shape: (batch_size, # of count answers)
                    gold_count_mask = (answer_as_counts != -1).long()
                    # Shape: (batch_size, # of count answers)
                    clamped_gold_counts = util.replace_masked_values(answer_as_counts, gold_count_mask, 0)
                    log_likelihood_for_counts = torch.gather(count_number_log_probs, 1, clamped_gold_counts)
                    # For those padded spans, we set their log probabilities to be very small negative value
                    log_likelihood_for_counts = util.replace_masked_values(log_likelihood_for_counts, gold_count_mask,
                                                                           -1e7)
                    # Shape: (batch_size, )
                    log_marginal_likelihood_for_count = util.logsumexp(log_likelihood_for_counts)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_count)
#                    print('log_marginal_likelihood_for_count: ', log_marginal_likelihood_for_count)
                elif answering_ability == "multiple_spans":
                    if self.multispan_head_name == "flexible_loss":
                        log_marginal_likelihood_for_multispan = \
                            self._multispan_log_likelihood(answer_as_text_to_disjoint_bios,
                                                           answer_as_list_of_bios,
                                                           span_bio_labels,
                                                           multispan_log_probs,
                                                           multispan_logits,
                                                           multispan_mask,
                                                           bio_wordpiece_mask,
                                                           is_bio_mask)
                    else:
                        log_marginal_likelihood_for_multispan = \
                            self._multispan_log_likelihood(span_bio_labels,
                                                           multispan_log_probs,
                                                           multispan_mask,
                                                           is_bio_mask,
                                                           logits=multispan_logits)
                    log_marginal_likelihood_list.append(log_marginal_likelihood_for_multispan)

                else:
                    raise ValueError(f"Unsupported answering ability: {answering_ability}")
            if len(self.answering_abilities) > 1:
                # Add the ability probabilities if there are more than one abilities
                all_log_marginal_likelihoods = torch.stack(log_marginal_likelihood_list, dim=-1)
                all_log_marginal_likelihoods = all_log_marginal_likelihoods + answer_ability_log_probs
                marginal_log_likelihood = util.logsumexp(all_log_marginal_likelihoods)
            else:
                marginal_log_likelihood = log_marginal_likelihood_list[0]
            output_dict["loss"] = - marginal_log_likelihood.mean()
#            print('batch_loss: ', output_dict["loss"])

        with torch.no_grad():
            best_answer_ability = best_answer_ability.detach().cpu().numpy()
            if metadata is not None:
                output_dict["question_id"] = []
                output_dict["answer"] = []
                i = 0
                while i < batch_size:
                    if len(self.answering_abilities) > 1:
                        answer_index = best_answer_ability[i]
                        predicted_ability_str = self.answering_abilities[answer_index]
                    else:
                        predicted_ability_str = self.answering_abilities[0]

                    answer_json: Dict[str, Any] = {}

                    question_start = 1
                    passage_start = len(metadata[i]["question_tokens"]) + 2
                    # We did not consider multi-mention answers here
                    if predicted_ability_str == "passage_span_extraction":
                        answer_json["answer_type"] = "passage_span"
                        passage_str = metadata[i]['original_passage']
                        offsets = metadata[i]['passage_token_offsets']
                        predicted_span = tuple(best_passage_span[i].detach().cpu().numpy())
                        start_offset = offsets[predicted_span[0] - passage_start][0]
                        end_offset = offsets[predicted_span[1] - passage_start][1]

                        end_offset = end_offset
                        predicted_answer = passage_str[start_offset:end_offset]
                        answer_json["value"] = predicted_answer
                        answer_json["spans"] = [(start_offset, end_offset)]

                    elif predicted_ability_str == "question_span_extraction":
                        answer_json["answer_type"] = "question_span"
                        question_str = metadata[i]['original_question']
                        offsets = metadata[i]['question_token_offsets']
                        predicted_span = tuple(best_question_span[i].detach().cpu().numpy())
                        start_offset = offsets[predicted_span[0] - question_start][0]
                        end_offset = offsets[predicted_span[1] - question_start][1]

                        end_offset = end_offset
                        predicted_answer = question_str[start_offset:end_offset]
                        answer_json["value"] = predicted_answer
                        answer_json["spans"] = [(start_offset, end_offset)]

                    elif predicted_ability_str == "addition_subtraction":  # plus_minus combination answer
                        answer_json["answer_type"] = "arithmetic"
                        original_numbers = metadata[i]['original_numbers']
                        sign_remap = {0: 0, 1: 1, 2: -1}
                        predicted_signs = [sign_remap[it] for it in best_signs_for_numbers[i].detach().cpu().numpy()]
                        result = sum([sign * number for sign, number in zip(predicted_signs, original_numbers)])
                        predicted_answer = format_number(result)

                        offsets = metadata[i]['passage_token_offsets']
                        number_indices = metadata[i]['number_indices']
                        number_positions = [offsets[index - 1] for index in number_indices]
                        answer_json['numbers'] = []
                        for offset, value, sign in zip(number_positions, original_numbers, predicted_signs):
                            answer_json['numbers'].append({'span': offset, 'value': value, 'sign': sign})
                        if number_indices[-1] == -1:
                            # There is a dummy 0 number at position -1 added in some cases; we are
                            # removing that here.
                            answer_json["numbers"].pop()
                        answer_json["value"] = result
                        answer_json['number_sign_log_probs'] = number_sign_log_probs[i, :, :].detach().cpu().numpy()
                    elif predicted_ability_str == "counting":
                        answer_json["answer_type"] = "count"
                        predicted_count = best_count_number[i].detach().cpu().numpy()
                        predicted_answer = str(predicted_count)
                        answer_json["count"] = predicted_count
                    elif predicted_ability_str == "multiple_spans":
                        passage_str = metadata[i]["original_passage"]
                        question_str = metadata[i]['original_question']
                        qp_tokens = metadata[i]["question_passage_tokens"]
                        answer_json["answer_type"] = "multiple_spans"
                        if self.multispan_head_name == "flexible_loss":
                            answer_json["value"], answer_json["spans"], invalid_spans = \
                                self._multispan_prediction(multispan_log_probs[i], multispan_logits[i], qp_tokens,
                                                           passage_str,
                                                           question_str,
                                                           multispan_mask[i], bio_wordpiece_mask[i],
                                                           self.multispan_use_prediction_beam_search and not self.training)
                        else:
                            answer_json["value"], answer_json["spans"], invalid_spans = \
                                self._multispan_prediction(multispan_log_probs[i], multispan_logits[i], qp_tokens,
                                                           passage_str,
                                                           question_str,
                                                           multispan_mask[i])
                        if self._unique_on_multispan:
                            answer_json["value"] = list(OrderedDict.fromkeys(answer_json["value"]))

                            if self._dont_add_substrings_to_ms:
                                answer_json["value"] = remove_substring_from_prediction(answer_json["value"])

                        if len(answer_json["value"]) == 0:
                            best_answer_ability[i] = top_two_answer_abilities[1][i][1]
                            continue
                        predicted_answer = answer_json["value"]
                    else:
                        raise ValueError(f"Unsupported answer ability: {predicted_ability_str}")

                    answer_json["predicted_answer"] = predicted_answer
                    output_dict["question_id"].append(metadata[i]["question_id"])
                    output_dict["answer"].append(answer_json)
                    answer_annotations = metadata[i].get('answer_annotations', [])
                    if answer_annotations:
                        self._drop_metrics(predicted_answer, answer_annotations)


                    ##################################################
                    real_answer_types=[]
                    answer_json['real']={}
                    if answer_as_passage_spans[i][0][0].detach().cpu().numpy() >= 0:#found passage span
                        real_answer_types.append('passage span')

                    if answer_as_question_spans[i][0][0].detach().cpu().numpy() >= 0:#found question span
                        real_answer_types.append('question span')

                    if answer_as_add_sub_expressions[i].sum().detach().cpu().numpy() > 0:#found expr
                        real_answer_types.append('expr')
                        expr_arr = answer_as_add_sub_expressions[i].detach().cpu().numpy()
                        sign_remap = {0: 0, 1: '', 2: '-'}
                        exprs = []
                        for j in range(min(5,len(expr_arr))):
                          parts=[]
                          for k in range(len(metadata[i]['original_numbers'])):
                            if expr_arr[j][k] > 0:
                              parts.append(str(sign_remap[expr_arr[j][k]])+str(metadata[i]['original_numbers'][k]))
                          exprs.append('+'.join(parts).replace('+-','-',1101))
                        answer_json['real']['exprs']=';'.join(exprs)
                    if answer_as_counts[i].detach().cpu().numpy()>0:
                        real_answer_types.append('count')
                    answer_json['real']['answer_types']=','.join(real_answer_types)
                    ##################################################

                    i += 1
            return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._drop_metrics.get_metric(reset)
        return {'em': exact_match, 'f1': f1_score}
