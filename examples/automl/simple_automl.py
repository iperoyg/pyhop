"""
A "toy example" for AutoML with HTN 
Author: Arthur Iperoyg <iperoyg#gmail.com>, May 02, 2019
"""
from pyhop import hop

def initial(state, p):
    return [('choosePP_task', p), ('setupClassifier_task', p), ('configPP_task', p)]
hop.declare_methods('initial_task',initial)

def setPP(state, pp):
    state.PP = pp
    return state

def choosePP_empty(state, p):
    return [('setPP', 'empty')]

def choosePP_PCA(state, p):
    return [('setPP', 'pca')]

hop.declare_methods('choosePP_task',choosePP_empty,choosePP_PCA)
hop.declare_operators(setPP)

def setClassifier(state, classifier):
    state.Classifier = classifier
    return state

def setParam(state, param, value):
    state.ClassifierParam[param] = value
    return state

def setupClassifier_basic(state, p):
    return [('setupClassifier_basic_task', p)]

def setupClassifier_basic_RandomForest(state, p):
    return [('setClassifier', 'random_forest'),('setParam','1',1)]

def setupClassifier_basic_C45(state, p):
    return [('setClassifier', 'c45'),('setParam','1',2)]

def setupClassifier_meta(state, p):
    return [('setupClassifier_meta_task', p)]

def setupClassifier_meta_Test(state, p):
    return [('setClassifier', 'test'),('setParam','1',0)]

hop.declare_methods('setupClassifier_task',setupClassifier_basic, setupClassifier_meta)
hop.declare_methods('setupClassifier_basic_task',setupClassifier_basic_RandomForest, setupClassifier_basic_C45)
hop.declare_methods('setupClassifier_meta_task',setupClassifier_meta_Test)
hop.declare_operators(setClassifier, setParam)

def setConfigPP(state, config):
    return state

def configPP_None(state, p):
    return [('setConfigPP', 1)]

hop.declare_methods('configPP_task',configPP_None)
hop.declare_operators(setConfigPP)

initialState = hop.State('initialState')
initialState.Name = "Teste"
initialState.ClassifierParam = dict()
initialState.TasksProbDistributions = dict()

#print('')
#hop.print_operators(hop.get_operators()) 
#print('')
#hop.print_methods(hop.get_methods())

methods = hop.get_methods() 
for t in methods:
    initialState.TasksProbDistributions[t] = [1 for x in methods[t]]

#print("\n\n")

#print(initialState.TasksProbDistributions)
plan = hop.plan(initialState,
         [('initial_task', '')],
         hop.get_operators(),
         hop.get_methods())

print(plan)