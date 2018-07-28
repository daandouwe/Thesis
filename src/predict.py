import os

from get_vocab import get_sentences

def predict(args, model, batches, name='test'):
    model.eval()
    sentences = get_sentences(os.path.join(args.data, 'test', 'ptb.{}.oracle'.format(name)))
    nsents = len(batches)
    for i, batch in enumerate(batches):
        sent, indices, actions = batch
        parser = model.parse(sent, indices)
        sentences[i]['actions'] = parser.actions
        if i % 100 == 0:
            print('Predicting: sentence {}/{}.'.format(i, nsents), end='\r')
    print()
    write_prediction(sentences, args.outdir, name='test')

def print_sent_dict_as_config(sent_dict, file):
    print(sent_dict['tree'], file=file)
    print(sent_dict['tags'], file=file)
    print(sent_dict['upper'], file=file)
    print(sent_dict['lower'], file=file)
    print(sent_dict['unked'], file=file)
    print('\n'.join(sent_dict['actions']), file=file)
    print(file=file)

def write_prediction(sentences, outdir, name, verbose=False):
    path = os.path.join(outdir, '{}.pred.oracle'.format(name))
    with open(path, 'w') as f:
        for i, sent_dict in enumerate(sentences):
            if verbose: print(i, end='\r')
            print_sent_dict_as_config(sent_dict, f)

def main():
    pass

if __name__ == '__main__':
    main()
