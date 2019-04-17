import os
import re
import subprocess

from utils.trees import fromstring, uncollapse


def evalb(evalb_dir, pred_path, gold_path, result_path, ignore_error=10000, param_file=None):
    """Use EVALB to score trees."""
    evalb_dir = os.path.expanduser(evalb_dir)

    assert os.path.exists(evalb_dir), f'do you have EVALB installed at {evalb_dir}?'


    evalb_exec = os.path.join(evalb_dir, "evalb")
    if param_file is None:
        command = '{} {} {} -e {} > {}'.format(
            evalb_exec,
            pred_path,
            gold_path,
            ignore_error,
            result_path
        )
    else:
        assert os.path.exists(param_file), f'cannot find parameter file at {param_file}?'
        command = '{} {} {} -e {} -p {} > {}'.format(
            evalb_exec,
            pred_path,
            gold_path,
            ignore_error,
            param_file,
            result_path
        )

    subprocess.run(command, shell=True)

    # Read result path and get F-sore.
    with open(result_path) as f:
        for line in f:
            match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore = float(match.group(1))
                return fscore

    return -1


class Score(object):
    """All information to compute the scores for a tree."""
    def __init__(self, num_words, num_correct, num_gold_spans, num_pred_spans):
        self.num_words = num_words
        self.num_correct = num_correct
        self.num_gold_spans = num_gold_spans
        self.num_pred_spans = num_pred_spans

    @property
    def precision(self):
        precision = self.num_correct / self.num_pred_spans
        return round(100 * precision, 2)

    @property
    def recall(self):
        recall = self.num_correct / self.num_gold_spans
        return round(100 * recall, 2)

    @property
    def fscore(self):
        precision = self.num_correct / self.num_pred_spans
        recall = self.num_correct / self.num_gold_spans
        if precision + recall == 0:
            return 0.0
        else:
            fscore = (2 * precision * recall) / (precision + recall)
            return round(100 * fscore, 2)


class Parseval(object):

    def __init__(self, gold_path, pred_path):
        self.gold_trees = self.load_trees(gold_path)
        self.pred_trees = self.load_trees(pred_path)

        assert len(self.gold_trees) == len(self.pred_trees)

    def load_trees(self, path):
        with open(path) as f:
            trees = [fromstring(line.strip()).convert() for line in f]
        return trees

    def check_trees(self, gold_tree, pred_tree):
        gold_words = gold_tree.words()
        pred_words = pred_tree.words()

        assert len(gold_words) == len(pred_words)
        assert all(w == v for w, v in zip(gold_words, pred_words))

    def score(self, gold_tree, pred_tree):
        gold_spans = uncollapse(gold_tree.spans())
        pred_spans = uncollapse(pred_tree.spans())
        correct = set(pred_spans) & set(gold_spans)
        score = Score(len(gold_tree.words()), len(correct), len(gold_spans), len(pred_spans))
        return score

    def recall(self, scores):
        num_correct = sum(score.num_correct for score in scores)
        num_gold_spans = sum(score.num_gold_spans for score in scores)
        recall = num_correct / num_gold_spans
        return round(100 * recall, 2)

    def precision(self, scores):
        num_correct = sum(score.num_correct for score in scores)
        num_pred_spans = sum(score.num_pred_spans for score in scores)
        precision = num_correct / num_pred_spans
        return round(100 * precision, 2)

    def fscore(self, scores):
        num_correct = sum(score.num_correct for score in scores)
        num_pred_spans = sum(score.num_pred_spans for score in scores)
        num_gold_spans = sum(score.num_gold_spans for score in scores)
        precision = num_correct / num_pred_spans
        recall = num_correct / num_gold_spans
        if precision + recall == 0:
            fscore = 0.0
        else:
            fscore = (2 * precision * recall) / (precision + recall)
        return round(100 * fscore, 2)

    def evaluate(self, tsv_output=False):
        scores = []
        for gold_tree, pred_tree in zip(self.gold_trees, self.pred_trees):
            self.check_trees(gold_tree, pred_tree)
            scores.append(self.score(gold_tree, pred_tree))

        if tsv_output:
            print('id', 'length', 'fscore', 'recall', 'precision', sep='\t')
            for i, score in enumerate(scores):
                print(i, score.num_words, score.fscore, score.recall, score.precision, sep='\t')

        else:
            print('ID     Len.     F1      Recal   Prec.')
            print('=====================================')
            for i, score in enumerate(scores):
                print(i, score.num_words, score.fscore, score.recall, score.precision, sep='\t')
            print('=====================================')
            print()
            print('=== Summary ===')
            print()
            print('Recall      =', self.recall(scores))
            print('Precision   =', self.precision(scores))
            print('Fscore      =', self.fscore(scores))
