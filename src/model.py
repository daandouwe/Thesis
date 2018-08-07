import torch
import torch.nn as nn
from torch.autograd import Variable

from data import PAD_INDEX, EMPTY_INDEX, REDUCED_INDEX, REDUCED_TOKEN, Item, wrap
from glove import load_glove
from embedding import ConvolutionalCharEmbedding
from nn import MLP
from encoder import StackLSTM, HistoryLSTM, BufferLSTM
from parser import Parser
from loss import LossCompute

class RNNG(nn.Module):
    """Recurrent Neural Network Grammar model."""
    def __init__(
        self,
        dictionary,
        word_embedding,
        nonterminal_embedding,
        action_embedding,
        buffer_encoder,
        stack_encoder,
        history_encoder,
        mlp,
        loss_compute,
        dropout,
        device
    ):
        super(RNNG, self).__init__()
        self.dictionary = dictionary
        self.device = device

        self.word_embedding = word_embedding
        self.nonterminal_embedding = nonterminal_embedding
        self.action_embedding = action_embedding

        # Actions
        # d = dictionary.n2i
        # self.SHIFT, self.REDUCE, self.OPEN = d['SHIFT'], d['REDUCE'], d['OPEN']
        # Parser encoders
        self.buffer_encoder = buffer_encoder
        self.stack_encoder = stack_encoder
        self.history_encoder = history_encoder

        # MLP for action classifiction
        self.mlp = mlp

        self.dropout = nn.Dropout(p=dropout)

        # Create an internal parser.
        self.parser = Parser(
            self.word_embedding,
            self.nonterminal_embedding,
            self.action_embedding,
            self.buffer_encoder,
            device=device
        )

        # Loss computation
        self.loss_compute = loss_compute

    def encode(self, stack, buffer, history):
        # Apply dropout.
        stack = self.dropout(stack)     # (batch, input_size)
        buffer = self.dropout(buffer)   # (batch, input_size)
        history = self.dropout(history) # (batch, input_size)
        # Encode
        b = buffer # buffer is already the top hidden state
        h = self.history_encoder(history) # returns top hidden state
        s = self.stack_encoder(stack) # returns top hidden state
        # Concatenate and apply mlp to obtain logits.
        x = torch.cat((b, h, s), dim=-1)
        logits = self.mlp(x)
        return logits

    def parse_step(self, action, logfile=False):
        """Updates parser one step give the action."""
        self.parser.history.push(action)

        # if action.token == self.SHIFT:
        if action.token == 'SHIFT':
            self.parser.shift()

        # elif action.token == self.REDUCE:
        elif action.token == 'REDUCE':
            # Pop all items from the open nonterminal.
            items, embeddings = self.parser.stack.pop()
            # Reduce these items using the composition function.
            x = self.stack_encoder.reduce(embeddings)
            # Push the new representation onto stack: the computed vector x
            # and a dummy index.
            self.parser.stack.push(
                Item(REDUCED_TOKEN, REDUCED_INDEX, embedding=x),
                reduced=True
            )

        # elif action.token == self.OPEN:
        elif action.token.startswith('NT'):
             # break relation with the original action Item
            item = Item(action.token, action.index)
            self.parser.stack.open_nonterminal(item)

        else:
            raise ValueError('got illegal action: {}'.format(a))

        # if logfile is not None:
        #     print(t, file=logfile)
        #     print(str(self.parser), file=logfile)
        #     print('Values : ', vals.numpy()[:10], file=logfile)
        #     print('Ids : ', ids.numpy()[:10], file=logfile)
        #     print('Action : ', action_id, action, file=logfile)
        #     print('Recalls : ', i, file=logfile)
        #     print(file=logfile)

    def forward(self, sentence, actions, logfile=None):
        """Forward training pass for RNNG.

        Arguments:
            sentence (list): input sentence as list of Item objects.
            actions (list): parse action sequence as list of Item objects.
        """
        # Initialize the parser with the sentence.
        self.parser.initialize(sentence)
        # Reset the hidden states of the StackLSTM and the HistoryLSTM.
        self.stack_encoder.initialize_hidden()
        self.history_encoder.initialize_hidden()
        # Cummulator variable for loss
        loss = Variable(torch.zeros(1, device=self.device))
        for t, action in enumerate(actions):
            # Compute parse representation and prediction.
            stack, buffer, history = self.parser.get_embedded_input()
            logits = self.encode(stack, buffer, history) # encode the parse configuration
            step_loss = self.loss_compute(logits, action.index)
            loss += step_loss
            # Take the appropriate parse step.
            self.parse_step(action)
        return loss


    def parse(self, sentence, logfile=None):
        """Parse an input sequence.

        Arguments:
            sentence (list): input sentence as list of Item objects.
        """
        # Initialize the parser with the sentence.
        self.parser.initialize(sentence)
        # Reset the hidden states of the StackLSTM and the HistoryLSTM.
        self.stack_encoder.initialize_hidden()
        self.history_encoder.initialize_hidden()
        t = 0
        while not self.parser.stack.empty:
            t += 1
            # Compute parse representation and prediction.
            stack, buffer, history = self.parser.get_embedded_input()
            logits = self.encode(stack, buffer, history) # encode the parse configuration
            # Get highest scoring valid predictions.
            vals, ids = logits.sort(descending=True)
            vals, ids = vals.data.squeeze(0), ids.data.squeeze(0)
            i = 0
            action = Item(self.dictionary.i2a[ids[i]], ids[i])
            while not self.parser.is_valid_action(action):
                i += 1
                action = Item(self.dictionary.i2a[ids[i]], action_id)
            # Take the appropriate parse step.
            self.parse_step(action)
        return self.parser


def make_model(args, dictionary):
    # Embeddings
    num_words = dictionary.num_words
    num_nonterminals = dictionary.num_nonterminals
    num_actions = dictionary.num_actions
    if args.use_char:
        word_embedding = ConvolutionalCharEmbedding(
                num_words, args.word_emb_dim, padding_idx=PAD_INDEX,
                dropout=args.dropout, device=args.device
            )
    # elif args.use_fasttext:
        # pass
    # elif args.use_elmo:
        # pass
    else:
        word_embedding = nn.Embedding(
                num_words, args.word_emb_dim, padding_idx=PAD_INDEX
            )
        if args.use_glove:
            words = [dictionary.i2w[i] for i in range(len(dictionary.w2i))]
            embeddings = load_glove(words, args.word_emb_dim, args.glove_dir, args.glove_error_dir)
            word_embedding.weight = nn.Parameter(embeddings)
            word_embedding.weight.requires_grad = False

    nonterminal_embedding = nn.Embedding(num_nonterminals, args.word_emb_dim, padding_idx=PAD_INDEX)
    action_embedding = nn.Embedding(num_actions, args.action_emb_dim, padding_idx=PAD_INDEX)

    # Encoders
    buffer_encoder = BufferLSTM(
        args.word_emb_dim,
        args.word_lstm_hidden,
        args.lstm_num_layers,
        args.dropout,
        args.device
    )
    stack_encoder = StackLSTM(
        args.word_emb_dim,
        args.word_lstm_hidden,
        args.dropout,
        args.device
    )
    history_encoder = HistoryLSTM(
        args.action_emb_dim,
        args.action_lstm_hidden,
        args.dropout,
        args.device
    )

    mlp_input = 2 * args.word_lstm_hidden + args.action_lstm_hidden
    mlp = MLP(
        mlp_input,
        args.mlp_dim,
        num_actions,
        args.dropout
    )

    loss_compute = LossCompute(nn.CrossEntropyLoss, args.device)

    model = RNNG(
        dictionary=dictionary,
        word_embedding=word_embedding,
        nonterminal_embedding=nonterminal_embedding,
        action_embedding=action_embedding,
        buffer_encoder=buffer_encoder,
        stack_encoder=stack_encoder,
        history_encoder=history_encoder,
        mlp=mlp,
        loss_compute=loss_compute,
        dropout=args.dropout,
        device=args.device
    )

    # Initialize parameters with Glorot.
    for p in model.parameters():
        if p.dim() > 1 and p.requires_grad:
            nn.init.xavier_uniform_(p)

    return model
