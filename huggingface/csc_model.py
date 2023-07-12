import json
import os
import shutil
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.file_download import http_user_agent
from torch import nn
from torch.nn import functional as F
from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import MaskedLMOutput, BaseModelOutputWithPooling
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertLMPredictionHead

cache_path = Path(os.path.abspath(__file__)).parent


def download_file(filename: str, path: Path):
    if os.path.exists(cache_path / filename):
        return

    if os.path.exists(path / filename):
        shutil.copyfile(path / filename, cache_path / filename)
        return

    hf_hub_download(
        "iioSnail/ChineseBERT-for-csc",
        filename,
        local_dir=cache_path,
        user_agent=http_user_agent(),
    )
    time.sleep(0.2)


class ChineseBertForCSC(BertPreTrainedModel):

    def __init__(self, config):
        super(ChineseBertForCSC, self).__init__(config)
        self.model = Dynamic_GlyceBertForMultiTask(config)
        self.tokenizer = None

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def _predict(self, sentence):
        if self.tokenizer is None:
            return "Please init tokenizer by `set_tokenizer(tokenizer)` before predict."

        inputs = self.tokenizer([sentence], return_tensors='pt')
        output_hidden = self.model(**inputs).logits
        return self.tokenizer.convert_ids_to_tokens(output_hidden.argmax(-1)[0, 1:-1])

    def predict(self, sentence, window=1):
        _src_tokens = list(sentence)
        src_tokens = list(sentence)
        pred_tokens = self._predict(sentence)

        for _ in range(window):
            record_index = []
            for i, (a, b) in enumerate(zip(src_tokens, pred_tokens)):
                if a != b:
                    record_index.append(i)

            src_tokens = pred_tokens
            pred_tokens = self._predict(''.join(pred_tokens))
            for i, (a, b) in enumerate(zip(src_tokens, pred_tokens)):
                # 若这个token被修改了，且在窗口范围内，则什么都不做。
                if a != b and any([abs(i - x) <= 1 for x in record_index]):
                    pass
                else:
                    pred_tokens[i] = src_tokens[i]

        return ''.join(pred_tokens)


#################################ChineseBERT Source Code##############################################
class Dynamic_GlyceBertForMultiTask(BertPreTrainedModel):
    def __init__(self, config):
        super(Dynamic_GlyceBertForMultiTask, self).__init__(config)

        self.bert = GlyceBertModel(config)
        self.cls = MultiTaskHeads(config)

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
            self,
            input_ids=None,
            pinyin_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs
    ):
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs_x = self.bert(
            input_ids,
            pinyin_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        encoded_x = outputs_x[0]

        prediction_scores = self.cls(encoded_x)

        return MaskedLMOutput(
            logits=prediction_scores,
            hidden_states=outputs_x.hidden_states,
            attentions=outputs_x.attentions,
        )


class GlyceBertModel(BertModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the models.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the models at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        models = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = models(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """

    def __init__(self, config):
        super(GlyceBertModel, self).__init__(config)
        self.config = config

        self.embeddings = FusionBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            pinyin_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the models is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the models is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, pinyin_ids=pinyin_ids, position_ids=position_ids, token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def forward_with_embedding(
            self,
            input_ids=None,
            pinyin_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            embedding=None
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the models is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the models is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        assert embedding is not None
        embedding_output = embedding
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class MultiTaskHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class FusionBertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position, glyph, pinyin and token_type embeddings.
    """

    def __init__(self, config):
        super(FusionBertEmbeddings, self).__init__()

        self.path = Path(config._name_or_path)
        config_path = cache_path / 'config'
        if not os.path.exists(config_path):
            os.makedirs(config_path)

        font_files = []
        download_file("config/STFANGSO.TTF24.npy", self.path)
        download_file("config/STXINGKA.TTF24.npy", self.path)
        download_file("config/方正古隶繁体.ttf24.npy", self.path)
        for file in os.listdir(config_path):
            if file.endswith(".npy"):
                font_files.append(config_path / file)
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.pinyin_embeddings = PinyinEmbedding(embedding_size=128, pinyin_out_dim=config.hidden_size, config=config)
        self.glyph_embeddings = GlyphEmbedding(font_npy_files=font_files)

        # self.LayerNorm is not snake-cased to stick with TensorFlow models variable name and be able to load
        # any TensorFlow checkpoint file
        self.glyph_map = nn.Linear(1728, config.hidden_size)
        self.map_fc = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, pinyin_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # get char embedding, pinyin embedding and glyph embedding
        word_embeddings = inputs_embeds  # [bs,l,hidden_size]
        pinyin_embeddings = self.pinyin_embeddings(pinyin_ids)  # [bs,l,hidden_size]
        glyph_embeddings = self.glyph_map(self.glyph_embeddings(input_ids))  # [bs,l,hidden_size]
        # fusion layer
        concat_embeddings = torch.cat((word_embeddings, pinyin_embeddings, glyph_embeddings), 2)
        inputs_embeds = self.map_fc(concat_embeddings)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PinyinEmbedding(nn.Module):

    def __init__(self, embedding_size: int, pinyin_out_dim: int, config):
        """
            Pinyin Embedding Module
        Args:
            embedding_size: the size of each embedding vector
            pinyin_out_dim: kernel number of conv
        """
        super(PinyinEmbedding, self).__init__()
        download_file("config/pinyin_map.json", Path(config._name_or_path))
        with open(cache_path / 'config' / 'pinyin_map.json') as fin:
            pinyin_dict = json.load(fin)
        self.pinyin_out_dim = pinyin_out_dim
        self.embedding = nn.Embedding(len(pinyin_dict['idx2char']), embedding_size)
        self.conv = nn.Conv1d(in_channels=embedding_size, out_channels=self.pinyin_out_dim, kernel_size=2,
                              stride=1, padding=0)

    def forward(self, pinyin_ids):
        """
        Args:
            pinyin_ids: (bs*sentence_length*pinyin_locs)

        Returns:
            pinyin_embed: (bs,sentence_length,pinyin_out_dim)
        """
        # input pinyin ids for 1-D conv
        embed = self.embedding(pinyin_ids)  # [bs,sentence_length,pinyin_locs,embed_size]
        bs, sentence_length, pinyin_locs, embed_size = embed.shape
        view_embed = embed.view(-1, pinyin_locs, embed_size)  # [(bs*sentence_length),pinyin_locs,embed_size]
        input_embed = view_embed.permute(0, 2, 1)  # [(bs*sentence_length), embed_size, pinyin_locs]
        # conv + max_pooling
        pinyin_conv = self.conv(input_embed)  # [(bs*sentence_length),pinyin_out_dim,H]
        pinyin_embed = F.max_pool1d(pinyin_conv, pinyin_conv.shape[-1])  # [(bs*sentence_length),pinyin_out_dim,1]
        return pinyin_embed.view(bs, sentence_length, self.pinyin_out_dim)  # [bs,sentence_length,pinyin_out_dim]


class GlyphEmbedding(nn.Module):
    """Glyph2Image Embedding"""

    def __init__(self, font_npy_files: List[str]):
        super(GlyphEmbedding, self).__init__()
        font_arrays = [
            np.load(np_file).astype(np.float32) for np_file in font_npy_files
        ]
        self.vocab_size = font_arrays[0].shape[0]
        self.font_num = len(font_arrays)
        self.font_size = font_arrays[0].shape[-1]
        # N, C, H, W
        font_array = np.stack(font_arrays, axis=1)
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.font_size ** 2 * self.font_num,
            _weight=torch.from_numpy(font_array.reshape([self.vocab_size, -1]))
        )

    def forward(self, input_ids):
        """
            get glyph images for batch inputs
        Args:
            input_ids: [batch, sentence_length]
        Returns:
            images: [batch, sentence_length, self.font_num*self.font_size*self.font_size]
        """
        # return self.embedding(input_ids).view([-1, self.font_num, self.font_size, self.font_size])
        return self.embedding(input_ids)
