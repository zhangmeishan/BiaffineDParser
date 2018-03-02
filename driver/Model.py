from driver.Layer import *
from data.Vocab import *


def drop_input_independent(word_embeddings, tag_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)
    word_masks = Variable(torch.bernoulli(word_masks), requires_grad=False)
    tag_masks = tag_embeddings.data.new(batch_size, seq_length).fill_(1 - dropout_emb)
    tag_masks = Variable(torch.bernoulli(tag_masks), requires_grad=False)
    scale = 3.0 / (2.0 * word_masks + tag_masks + 1e-12)
    word_masks *= scale
    tag_masks *= scale
    word_masks = word_masks.unsqueeze(dim=2)
    tag_masks = tag_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks
    tag_embeddings = tag_embeddings * tag_masks

    return word_embeddings, tag_embeddings

def drop_sequence_sharedmask(inputs, dropout, batch_first=True):
    if batch_first:
        inputs = inputs.transpose(0, 1)
    seq_length, batch_size, hidden_size = inputs.size()
    drop_masks = inputs.data.new(batch_size, hidden_size).fill_(1 - dropout)
    drop_masks = Variable(torch.bernoulli(drop_masks), requires_grad=False)
    drop_masks = drop_masks / (1 - dropout)
    drop_masks = torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, seq_length).permute(2, 0, 1)
    inputs = inputs * drop_masks

    return inputs.transpose(1, 0)


class ParserModel(nn.Module):
    def __init__(self, vocab, config, pretrained_embedding):
        super(ParserModel, self).__init__()
        self.config = config
        self.word_embed = nn.Embedding(vocab.vocab_size, config.word_dims, padding_idx=0)
        self.extword_embed = nn.Embedding(vocab.extvocab_size, config.word_dims, padding_idx=0)
        self.tag_embed = nn.Embedding(vocab.tag_size, config.tag_dims, padding_idx=0)

        word_init = np.zeros((vocab.vocab_size, config.word_dims), dtype=np.float32)
        self.word_embed.weight.data.copy_(torch.from_numpy(word_init))
        
        tag_init = np.random.randn(vocab.tag_size, config.tag_dims).astype(np.float32)
        self.tag_embed.weight.data.copy_(torch.from_numpy(tag_init))

        self.extword_embed.weight.data.copy_(torch.from_numpy(pretrained_embedding))
        self.extword_embed.weight.requires_grad = False


        self.lstm = MyLSTM(
            input_size=config.word_dims + config.tag_dims,
            hidden_size=config.lstm_hiddens,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout_in = config.dropout_lstm_input,
            dropout_out=config.dropout_lstm_hidden,
        )

        self.mlp_arc_dep = NonLinear(
            input_size = 2*config.lstm_hiddens,
            hidden_size = config.mlp_arc_size+config.mlp_rel_size,
            activation = nn.LeakyReLU(0.1))
        self.mlp_arc_head = NonLinear(
            input_size = 2*config.lstm_hiddens,
            hidden_size = config.mlp_arc_size+config.mlp_rel_size,
            activation = nn.LeakyReLU(0.1))

        self.total_num = int((config.mlp_arc_size+config.mlp_rel_size) / 100)
        self.arc_num = int(config.mlp_arc_size / 100)
        self.rel_num = int(config.mlp_rel_size / 100)

        self.arc_biaffine = Biaffine(config.mlp_arc_size, config.mlp_arc_size, \
                                     1, bias=(True, False))
        self.rel_biaffine = Biaffine(config.mlp_rel_size, config.mlp_rel_size, \
                                     vocab.rel_size, bias=(True, True))

    def forward(self, words, extwords, tags, masks):
        # x = (batch size, sequence length, dimension of embedding)
        x_word_embed = self.word_embed(words)
        x_extword_embed = self.extword_embed(extwords)
        x_embed = x_word_embed + x_extword_embed
        x_tag_embed = self.tag_embed(tags)

        if self.training:
            x_embed, x_tag_embed = drop_input_independent(x_embed, x_tag_embed, self.config.dropout_emb)

        x_lexical = torch.cat((x_embed, x_tag_embed), dim=2)

        outputs, _ = self.lstm(x_lexical, masks, None)
        outputs = outputs.transpose(1, 0)

        if self.training:
            outputs = drop_sequence_sharedmask(outputs, self.config.dropout_mlp)

        x_all_dep = self.mlp_arc_dep(outputs)
        x_all_head = self.mlp_arc_head(outputs)

        if self.training:
            x_all_dep = drop_sequence_sharedmask(x_all_dep, self.config.dropout_mlp)
            x_all_head = drop_sequence_sharedmask(x_all_head, self.config.dropout_mlp)

        x_all_dep_splits = torch.split(x_all_dep, split_size=100, dim=2)
        x_all_head_splits = torch.split(x_all_head, split_size=100, dim=2)

        x_arc_dep = torch.cat(x_all_dep_splits[:self.arc_num], dim=2)
        x_arc_head = torch.cat(x_all_head_splits[:self.arc_num], dim=2)

        arc_logit = self.arc_biaffine(x_arc_dep, x_arc_head)
        arc_logit = torch.squeeze(arc_logit, dim=3)

        x_rel_dep = torch.cat(x_all_dep_splits[self.arc_num:], dim=2)
        x_rel_head = torch.cat(x_all_head_splits[self.arc_num:], dim=2)

        rel_logit_cond = self.rel_biaffine(x_rel_dep, x_rel_head)
        return arc_logit, rel_logit_cond