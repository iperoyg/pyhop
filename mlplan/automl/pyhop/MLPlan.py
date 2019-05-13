from .htn import HTN, Goal, State

class MLPlan():
    def __init__(self):
        self.hop = HTN()
        self.hop.declare_methods('initial_task', self.initial)
        self.hop.declare_methods('choosePP_task',self.choosePP_empty,self.choosePP_PCA)
        self.hop.declare_operators(self.setPP)
        self.hop.declare_methods('setupClassifier_task',self.setupClassifier_basic, self.setupClassifier_meta)
        self.hop.declare_methods('setupClassifier_basic_task',self.setupClassifier_basic_RandomForest, self.setupClassifier_basic_C45)
        self.hop.declare_methods('setupClassifier_meta_task',self.setupClassifier_meta_Test)
        self.hop.declare_operators(self.setClassifier, self.setParam)
        self.hop.declare_methods('configPP_task',self.configPP_None)
        self.hop.declare_operators(self.setConfigPP)

        self.initialState = State('initialState')
        self.initialState.Name = "Teste"
        self.initialState.ClassifierParam = dict()
        self.initialState.TasksProbDistributions = dict()
        methods = self.hop.get_methods() 
        for t in methods:
            self.initialState.TasksProbDistributions[t] = [1 for x in methods[t]]
        
    def plan(self):
        return self.hop.plan(self.initialState,
            [('initial_task', '')],
            self.hop.get_operators(),
            self.hop.get_methods())

    def initial(self,state, p):
        return [('choosePP_task', p), ('setupClassifier_task', p), ('configPP_task', p)]
    
    def setPP(self, state, pp):
        state.PP = pp
        return state

    def choosePP_empty(self, state, p):
        return [('setPP', 'empty')]

    def choosePP_PCA(self, state, p):
        return [('setPP', 'pca')]

    def setClassifier(self, state, classifier):
        state.Classifier = classifier
        return state

    def setParam(self, state, param, value):
        state.ClassifierParam[param] = value
        return state

    def setupClassifier_basic(self, state, p):
        return [('setupClassifier_basic_task', p)]

    def setupClassifier_basic_RandomForest(self, state, p):
        return [('setClassifier', 'random_forest'),('setParam','1',1)]

    def setupClassifier_basic_C45(self, state, p):
        return [('setClassifier', 'c45'),('setParam','1',2),('setParam','fruta','banana')]

    def setupClassifier_meta(self, state, p):
        return [('setupClassifier_meta_task', p)]

    def setupClassifier_meta_Test(self, state, p):
        return [('setClassifier', 'test'),('setParam','1',0)]

    def setConfigPP(self, state, config):
        return state

    def configPP_None(self, state, p):
        return [('setConfigPP', 1)]

    
