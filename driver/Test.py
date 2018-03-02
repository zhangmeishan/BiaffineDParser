import sys
sys.path.extend(["../../","../","./"])
import time
import numpy as np
import pickle
import argparse
from driver.Config import *
from driver.Model import *
from driver.Parser import *
from data.Dataloader import *
import pickle
import random

def evaluate(data, parser, vocab, outputFile):
    start = time.time()
    parser.model.eval()
    output = open(outputFile, 'w', encoding='utf-8')
    arc_total_test, arc_correct_test, rel_total_test, rel_correct_test = 0, 0, 0, 0

    for onebatch in data_iter(data, config.test_batch_size, False):
        words, extwords, tags, heads, rels, lengths, masks = \
            batch_data_variable(onebatch, vocab)
        count = 0
        arcs_batch, rels_batch = parser.parse(words, extwords, tags, lengths, masks)
        for tree in batch_variable_depTree(onebatch, arcs_batch, rels_batch, lengths, vocab):
            printDepTree(output, tree)
            arc_total, arc_correct, rel_total, rel_correct = evalDepTree(tree, onebatch[count])
            arc_total_test += arc_total
            arc_correct_test += arc_correct
            rel_total_test += rel_total
            rel_correct_test += rel_correct
            count += 1

    output.close()
    uas = arc_correct_test * 100.0 / arc_total_test
    las = rel_correct_test * 100.0 / rel_total_test


    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  parser time = %.2f " % (len(data), during_time))

    return arc_correct_test, rel_correct_test, arc_total_test, uas, las


if __name__ == '__main__':
    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='examples/default.cfg')
    argparser.add_argument('--model', default='BaseParser')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    vocab = pickle.load(open(config.load_vocab_path, 'rb'))
    vec = vocab.create_pretrained_embs(config.pretrained_embeddings_file)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    # print(config.use_cuda)

    model = ParserModel(vocab, config, vec)
    model.load_state_dict(torch.load(config.load_model_path))
    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        model = model.cuda()

    parser = BiaffineParser(model, vocab.ROOT)
    test_data = read_corpus(config.test_file, vocab)

    arc_correct, rel_correct, arc_total, test_uas, test_las = \
        evaluate(test_data, parser, vocab, config.test_file + '.out')
    print("Test: uas = %d/%d = %.2f, las = %d/%d =%.2f" % \
          (arc_correct, arc_total, test_uas, rel_correct, arc_total, test_las))


