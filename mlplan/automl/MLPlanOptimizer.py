import argparse
import json
from base_optimizer import BaseOptimizer
from pyhop.MLPlan import MLPlan
from parsers.candidate_parser import *

class MLPLanOptimizer(BaseOptimizer):
    def __init__(self, server_url, dataset, metrics_list, splits, grammar_file, number_of_evaluations, seed):
        self.number_of_evaluations = number_of_evaluations
        self.optimizer = MLPlan()
        super(MLPLanOptimizer, self).__init__(
            server_url, dataset, metrics_list, splits, grammar_file, seed)

    def optimize(self):
        #pop = candidate_solution.get_random_candidates(
        #    self.grammar_file, self.number_of_evaluations, self.seed)
        dictionary = dict()
        print('------------------------------------')
        print("Evaluating the pipelines....")
        print('------------------------------------')
        
        for i in range(3):
            try:
                plan = self.optimizer.plan()
                print('plano', plan)
                candidate = self.parser_plan_to_pipeline(plan)
                #print(candidate)
                results = self.evaluate_pipeline(candidate)
                #print(results)
                print("#Pipeline: " + str(i))
               # print("#Pipeline's parse tree: " + str(pip[2]))
                print("#Evaluation performance (F1 weighted): " + str(results['f1']['mean']))
                dictionary[i] = results['f1']['mean']
            except Exception as e:
                print("#" + str(e))
            
            print()

        sorted_dictionary = list(
            sorted(dictionary.items(), key=lambda kv: kv[1]))

        return sorted_dictionary
    
    def parser_plan_to_pipeline(self, plan):
        plan_as_string = "LDA 5 0.001"
        return load_pipeline(plan_as_string)

def parse_args():
    parser = argparse.ArgumentParser(
        description='TODO')

    # Arguments that are necessary for all optimizer.
    parser.add_argument('-d', '--dataset',
                        required=True, type=str,
                        help='Name of the dataset.')

    parser.add_argument('-p', '--optimizer_config',
                        type=str,
                        help='File that configures the optimizer.')

    parser.add_argument('-s', '--server_config',
                        type=str,
                        help='File that configures the connection with '
                             'the server.')
    parser.add_argument('-g', '--grammar_file',
                        required=True, type=str,
                        help='File that contains the grammar.')
    parser.add_argument('-seed', '--seed',
                        type=int, default=0,
                        help='Seed to control the generation of '
                             'pseudo-random numbers.')

    # This argument is specific for this optimizer.
    # You can define your own stopping criterion.
    parser.add_argument('-n', '--number_of_evaluations',
                        type=int, default=5,
                        help='Number of pipeline evaluations considered in '
                             'the random search.')

    return parser.parse_args()


def main(args):
    server_url = 'http://automl.speed.dcc.ufmg.br:80'

    if args.server_config is not None:
        server_url = json.load(open(args.server_config))['serverUrl']

    metrics_list = [{
        'metric': 'f1',
        'args': {'average': 'micro'},
        'name': 'f1_score'
    }]

    splits = 5

    if args.optimizer_config is not None:
        config = json.load(open(args.optimizer_config))
        metrics_list = config['metrics']
        splits = config['splits']

    print('----- RPC Client configuration -----')
    print('Server url:', server_url)
    print('\nDataset:', args.dataset)
    print('\nMetrics:', metrics_list)
    print('\nSplits:', splits)


    rand_opt = MLPLanOptimizer(server_url, args.dataset, metrics_list, splits,
                         args.grammar_file, args.number_of_evaluations, args.seed)
    rand_result = rand_opt.optimize()
    print(rand_result)
    rs_length = len(rand_result)

    print('\n------------------------------------')
    key_best = rand_result[rs_length - 1][0]
    value_best = rand_result[rs_length - 1][1]
    print('Best pipeline: ' + str(key_best))
    print('Best pipeline\'s result (F1 weighted): ' + str(value_best))


if __name__ == '__main__':
    args = parse_args()

    main(args)