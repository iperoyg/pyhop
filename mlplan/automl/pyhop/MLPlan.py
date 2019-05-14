from .htn import HTN, Goal, State

class MLPlan():
    def __init__(self):
        self.hop = HTN()
        self.hop.declare_methods('initial_task', self.initial)
        self.hop.declare_methods('choosePP_task',self.choosePP_empty,self.choosePP_kBest)
        self.hop.declare_operators(self.setPP)
        self.hop.declare_methods('setupClassifier_task',self.setupClassifier_basic)
        self.hop.declare_methods('setupClassifier_basic_task', self.setupClassifier_basic_LDA)
        self.hop.declare_methods('setupClassifier_meta_task',self.setupClassifier_meta_Test)
        self.hop.declare_operators(self.setClassifier, self.setParam)
        self.hop.declare_methods('configPP_task',self.configPP_None)
        self.hop.declare_operators(self.setConfigPP)

        #kbest
        self.hop.declare_methods('config_kBest_k_task', self.set_kBest_k_1, self.set_kBest_k_2, self.set_kBest_k_3)
        #kbest

        #lda
        self.hop.declare_methods('config_LDA_NC_task', self.set_LDA_NC_1, self.set_LDA_NC_2)
        self.hop.declare_methods('config_LDA_tol_task', self.set_LDA_tol_0001, self.set_LDA_tol_001)
        #lda

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

    #kbest
    def choosePP_kBest(self, state, p):
        return [('setPP', 'kBest'),('config_kBest_k_task', p)]
    def set_kBest_k_1(self, state, p):
        return [('setParam', 'k', 1)]
    def set_kBest_k_2(self, state, p):
        return [('setParam', 'k', 2)]
    def set_kBest_k_3(self, state, p):
        return [('setParam', 'k', 3)]
    #kbest
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

    #lda
    def setupClassifier_basic_LDA(self, state, p):
        return [('setClassifier', 'LDA'),('config_LDA_NC_task',p),('config_LDA_tol_task',p)]
    def set_LDA_NC_1(self, state, p):
        return [('setParam', 'n_components', 1)]
    def set_LDA_NC_2(self, state, p):
        return [('setParam', 'n_components', 2)]
    def set_LDA_tol_0001(self, state, p):
        return [('setParam', 'tol', 0.0001)]
    def set_LDA_tol_001(self, state, p):
        return [('setParam', 'tol', 0.001)]
    #lda

    def setupClassifier_meta(self, state, p):
        return [('setupClassifier_meta_task', p)]

    def setupClassifier_meta_Test(self, state, p):
        return [('setClassifier', 'test'),('setParam','1',0)]

    def setConfigPP(self, state, config):
        return state

    def configPP_None(self, state, p):
        return [('setConfigPP', 1)]

    
