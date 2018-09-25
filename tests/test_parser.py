import argparse

from parser import Parser
from data import PAD_TOKEN, PAD_INDEX
from encoder import BiRecurrentEncoder, StackLSTM, BufferLSTM, HistoryLSTM
from nn import MLP
from loss import LossCompute


LONG = {
    tagged_tree : '(S (NP (NNP Avco) (NNP Corp.)) (VP (VBD received) (NP (NP (DT an) (ADJP (QP ($ $) (CD 11.8) (CD million))) (NNP Army) (NN contract)) (PP (IN for) (NP (NN helicopter) (NNS engines))))) (. .))',
    tree : '(S (NP Avco Corp.) (VP received (NP (NP an (ADJP (QP $ 11.8 million)) Army contract) (PP for (NP helicopter engines)))) .)',
    sentence : 'Avco Corp. received an $ 11.8 million Army contract for helicopter engines .'.split(),
    actions : [
        'NT(S)',
        'NT(NP)',
        'SHIFT',
        'SHIFT',
        'REDUCE',
        'NT(VP)',
        'SHIFT',
        'NT(NP)',
        'NT(NP)',
        'SHIFT',
        'NT(ADJP)',
        'NT(QP)',
        'SHIFT',
        'SHIFT',
        'SHIFT',
        'REDUCE',
        'REDUCE',
        'SHIFT',
        'SHIFT',
        'REDUCE',
        'NT(PP)',
        'SHIFT',
        'NT(NP)',
        'SHIFT',
        'SHIFT',
        'REDUCE',
        'REDUCE',
        'REDUCE',
        'REDUCE',
        'SHIFT',
        'REDUCE'
    ]
}

SHORT = {
    tagged_tree : "(S (NP (NN Champagne) (CC and) (NN dessert)) (VP (VBD followed)) (. .))",
    tree : "(S (NP Champagne and dessert) (VP followed) .)",
    sentence : "Champagne and dessert followed .".split(),
    actions : [
        'NT(S)',
        'NT(NP)',
        'SHIFT',
        'SHIFT',
        'SHIFT',
        'REDUCE',
        'NT(VP)',
        'SHIFT',
        'REDUCE',
        'SHIFT',
        'REDUCE'
    ]
}


def prepare_data(actions, sentence):
    i2n = [PAD_TOKEN] + [a[3:-1] for a in actions if a.startswith('NT')]
    i2w = [w for w in list(set(sentence))]
    n2i = dict((n, i) for i, n in enumerate(i2n))
    w2i = dict((w, i) for i, w in enumerate(i2w))
    action_items = []
    sentence_items = []
    for token in sentence:
        index = w2i[token]
        sentence_items.append(Word(token, index))
    for token in actions:
        if token == SHIFT.token:
            action = SHIFT
        elif token == REDUCE.token:
            action = REDUCE
        elif token.startswith('NT(') and token.endswith(')'):
            nt = token[3:-1]
            nt = Nonterminal(nt, n2i[nt])
            action = NT(nt)
        elif token.startswith('GEN(') and token.endswith(')'):
            word = token[4:-1]
            word = Word(word, w2i[word])
            action = GEN(word)
        else:
            raise ValueError(f'found strange token: {token}')
        action_items.append(action)
    return action_items, sentence_items, len(i2w), len(i2n)


def test_parser(actions, sentence, tree, dim=4):
    assert dim % 2 == 0
    actions, sentence, num_words, num_nonterm = prepare_data(actions, sentence)
    word_embedding = nn.Embedding(num_words, dim)
    nt_embedding = nn.Embedding(num_nonterm, dim)
    action_embedding = nn.Embedding(3, dim)

    stack_encoder = StackLSTM(dim, dim, dropout=0.3)
    buffer_encoder = BufferLSTM(dim, dim, 2, dropout=0.3)
    history_encoder = HistoryLSTM(dim, dim, dropout=0.3)
    parser = Parser(
        word_embedding,
        nt_embedding,
        action_embedding,
        stack_encoder,
        buffer_encoder,
        history_encoder,
    )
    parser.eval()
    parser.initialize(sentence)
    for i, action in enumerate(actions):
        print('--------')
        print(f'Step {i:>3}')
        print(parser.stack)
        print(parser.buffer)
        print(parser.history)
        print('action: {}'.format(action.token))
        parser.parse_step(action)
        if i > 0:
            print('partial tree: {}'.format(parser.stack.get_tree(with_tag=False)))
            print('')
    print('--------')
    print('Finished')
    print(parser.stack)
    print(parser.buffer)
    print(parser.history)
    print(f'open nonterminals: {parser.stack.num_open_nonterminals}')
    print('')
    print('pred: {}'.format(parser.stack.get_tree(with_tag=False)))
    print('gold: {}'.format(tree))


def forward(model, actions, sentence):
    parser, action_mlp, nonterminal_mlp = model
    parser.initialize(sentence)
    loss_compute = LossCompute(nn.CrossEntropyLoss, device=None)
    loss = torch.zeros(1)
    for i, action in enumerate(actions):
        # Compute loss
        stack, buffer, history = parser.get_encoded_input()
        x = torch.cat((buffer, history, stack), dim=-1)
        action_logits = action_mlp(x)
        loss += loss_compute(action_logits, action.action_index)
        # If we open a nonterminal, predict which.
        if action.is_nt:
            nonterminal_logits = nonterminal_mlp(x)
            nt = action.get_nt()
            loss += loss_compute(nonterminal_logits, nt.index)
        parser.parse_step(action)
    return loss


def test_train(actions, sentence, steps, dim=4):
    assert dim % 2 == 0
    num_actions = 3
    data = prepare_data(actions, sentence)
    actions, sentence, num_words, num_nonterm = data

    word_embedding = nn.Embedding(num_words, dim)
    nt_embedding = nn.Embedding(num_nonterm, dim)
    action_embedding = nn.Embedding(num_actions, dim)

    stack_encoder = StackLSTM(dim, dim, dropout=0.3)
    buffer_encoder = BufferLSTM(dim, dim, 2, dropout=0.3)
    history_encoder = HistoryLSTM(dim, dim, dropout=0.3)
    parser = Parser(
        word_embedding,
        nt_embedding,
        action_embedding,
        stack_encoder,
        buffer_encoder,
        history_encoder,
    )
    action_mlp = MLP(3*dim, dim, num_actions)
    nonterminal_mlp = MLP(3*dim, dim, num_nonterm)

    parameters = (
        list(parser.parameters()) +
        list(action_mlp.parameters()) +
        list(nonterminal_mlp.parameters())
    )
    optimizer = torch.optim.Adam(parameters, lr=0.001)
    model = (parser, action_mlp, nonterminal_mlp)


    for i in range(steps):
        loss = forward(model, actions, sentence)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('loss', loss.item(), end='\r')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--long', action='store_true')
    args = parser.parse_args()

    if args.long:
        sentence = LONG['sentence']
        actions = LONG['actions']
        tree = LONG['tree']
    else:
        sentence = SHORT['sentence']
        actions = SHORT['actions']
        tree = SHORT['tree']

    test_parser(actions, sentence, tree)
    # test_train(actions, sentence, dim=50, steps=1000)
