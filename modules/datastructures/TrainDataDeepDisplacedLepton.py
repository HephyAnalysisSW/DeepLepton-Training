from DeepJetCore.TrainData import TrainData
from DeepJetCore.TrainData import fileTimeOut as tdfto
import numpy as np
import os
# set tree name to use
import DeepJetCore.preprocessing
DeepJetCore.preprocessing.setTreeName('tree')


def fileTimeOut(fileName, timeOut):
    tdfto(fileName, timeOut)


class TrainDataDeepDisplacedLepton(TrainData):
    '''
    Base class for DeepLepton Training Data.
    '''

    def __init__(self):
        TrainData.__init__(self)

        self.description = "DeepLepton training datastructure"
        self.truth_branches = ['lep_isPromptId_Training',
                               'lep_isNonPromptId_Training',
                               'lep_isFakeId_Training',
                               'lep_isFromSUSY_Training',
                               'lep_isFromSUSYHF_Training',]  # 'lep_isFromSUSYHF_Training'
        self.undefTruth = []
        self.weightbranchX = 'lep_pt'
        self.weightbranchY = 'lep_dxy'
        self.remove = False
        self.referenceclass = 'lep_isFromSUSY_Training' # later maybe lep_isFromSUSYandHF_Training
        # setting DeepLepton specific defaults
        self.treename = "tree"
        self.firstfilepred = True
        # self.undefTruth=['isUndefined']
        #self.red_classes      = ['cat_P', 'cat_NP', 'cat_F']
        #self.reduce_truth     = ['lep_isPromptId_Training', 'lep_isNonPromptId_Training', 'lep_isFakeId_Training']
        #self.class_weights    = [1.00, 1.00, 1.00]

        self.weight_binX = np.array([
            5, 7.5, 10, 12.5, 15, 17.5, 20, 25, 30, 35, 40, 45, 50, 60, 75, 100,
            125, 150, 175, 200, 250, 300, 400, 500,
            600, 2000], dtype=float)
        #self.weight_binX = np.geomspace(3.5, 2000, 30)

        # for lep_eta it was:
        # self.weight_binY = np.array(
        #     [-2.5, -2., -1.5, -1., -0.5, 0.5, 1, 1.5, 2., 2.5],
        #     dtype=float
        # )
        
        # consicer num=20 or higher?
        self.weight_binY = np.linspace(start=-5, stop=5, num=10, endpoint=True, dtype=float)
        
        # if training not flat in pt, eta, ... maybe remove them..
        self.global_branches = [
            'lep_pt',                   # 0 
            'lep_eta',                  # 1
            'lep_phi',                  # 2
            'lep_mediumId',             # 3
            'lep_miniPFRelIso_all',     # 4
            'lep_sip3d',                # 5
            'lep_dxy',                  # 6
            'lep_dz',
            'lep_charge',
            'lep_dxyErr',
            'lep_dzErr',
            'lep_ip3d',
            'lep_jetPtRelv2',
            'lep_jetRelIso',
            'lep_miniPFRelIso_chg',
            'lep_mvaLowPt',
            'lep_nStations',
            'lep_nTrackerLayers',
            'lep_pfRelIso03_all',
            'lep_pfRelIso03_chg',
            'lep_pfRelIso04_all',
            'lep_ptErr',
            'lep_segmentComp',
            'lep_tkRelIso',
            'lep_tunepRelPt', ]

        self.pfCand_neutral_branches = ['pfCand_neutral_eta',
                                        'pfCand_neutral_phi',
                                        'pfCand_neutral_pt',
                                        'pfCand_neutral_puppiWeight',
                                        'pfCand_neutral_puppiWeightNoLep',
                                        'pfCand_neutral_ptRel',
                                        'pfCand_neutral_deltaR', ]
        self.npfCand_neutral = 10

        self.pfCand_charged_branches = ['pfCand_charged_d0',
                                        'pfCand_charged_d0Err',
                                        'pfCand_charged_dz',
                                        'pfCand_charged_dzErr',
                                        'pfCand_charged_eta',
                                        'pfCand_charged_mass',
                                        'pfCand_charged_phi',
                                        'pfCand_charged_pt',
                                        'pfCand_charged_puppiWeight',
                                        'pfCand_charged_puppiWeightNoLep',
                                        'pfCand_charged_trkChi2',
                                        'pfCand_charged_vtxChi2',
                                        'pfCand_charged_charge',
                                        'pfCand_charged_lostInnerHits',
                                        'pfCand_charged_pvAssocQuality',
                                        'pfCand_charged_trkQuality',
                                        'pfCand_charged_ptRel',
                                        'pfCand_charged_deltaR', ]
        self.npfCand_charged = 80

        self.pfCand_photon_branches = ['pfCand_photon_eta',
                                       'pfCand_photon_phi',
                                       'pfCand_photon_pt',
                                       'pfCand_photon_puppiWeight',
                                       'pfCand_photon_puppiWeightNoLep',
                                       'pfCand_photon_ptRel',
                                       'pfCand_photon_deltaR', ]
        self.npfCand_photon = 50

        self.pfCand_electron_branches = ['pfCand_electron_d0',
                                         'pfCand_electron_d0Err',
                                         'pfCand_electron_dz',
                                         'pfCand_electron_dzErr',
                                         'pfCand_electron_eta',
                                         'pfCand_electron_mass',
                                         'pfCand_electron_phi',
                                         'pfCand_electron_pt',
                                         'pfCand_electron_puppiWeight',
                                         'pfCand_electron_puppiWeightNoLep',
                                         'pfCand_electron_trkChi2',
                                         'pfCand_electron_vtxChi2',
                                         'pfCand_electron_charge',
                                         'pfCand_electron_lostInnerHits',
                                         'pfCand_electron_pvAssocQuality',
                                         'pfCand_electron_trkQuality',
                                         'pfCand_electron_ptRel',
                                         'pfCand_electron_deltaR', ]
        self.npfCand_electron = 4

        self.pfCand_muon_branches = ['pfCand_muon_d0',
                                     'pfCand_muon_d0Err',
                                     'pfCand_muon_dz',
                                     'pfCand_muon_dzErr',
                                     'pfCand_muon_eta',
                                     'pfCand_muon_mass',
                                     'pfCand_muon_phi',
                                     'pfCand_muon_pt',
                                     'pfCand_muon_puppiWeight',
                                     'pfCand_muon_puppiWeightNoLep',
                                     'pfCand_muon_trkChi2',
                                     'pfCand_muon_vtxChi2',
                                     'pfCand_muon_charge',
                                     'pfCand_muon_lostInnerHits',
                                     'pfCand_muon_pvAssocQuality',
                                     'pfCand_muon_trkQuality',
                                     'pfCand_muon_ptRel',
                                     'pfCand_muon_deltaR']
        self.npfCand_muon = 6

        self.SV_branches = ['SV_dlen',
                            'SV_dlenSig',
                            'SV_dxy',
                            'SV_dxySig',
                            'SV_pAngle',
                            'SV_chi2',
                            'SV_eta',
                            'SV_mass',
                            'SV_ndof',
                            'SV_phi',
                            'SV_pt',
                            'SV_x',
                            'SV_y',
                            'SV_z',
                            'SV_ptRel',
                            'SV_deltaR', ]
        self.nSV = 10

    def createWeighterObjects(self, allsourcefiles):
        #
        # Calculates the weights needed for flattening the pt/eta spectrum

        from DeepJetCore.Weighter import Weighter
        weighter = Weighter()
        weighter.undefTruth = self.undefTruth
        branches = [self.weightbranchX, self.weightbranchY]
        branches.extend(self.truth_branches)

        if self.remove:
            weighter.setBinningAndClasses(
                [self.weight_binX,  self.weight_binY],
                self.weightbranchX, self.weightbranchY,
                self.truth_branches,
                method=self.referenceclass
            )

        counter = 0
        import ROOT
        from root_numpy import tree2array, root2array
        if self.remove:
            for fname in allsourcefiles:
                fileTimeOut(fname, 120)
                nparray = root2array(
                    fname,
                    treename="tree",
                    stop=None,
                    branches=branches
                )
                norm_hist = True
                if self.referenceclass == 'flatten':
                    norm_hist = False
                weighter.addDistributions(nparray, norm_h=norm_hist)
                #del nparray
                counter = counter+1
            weighter.createRemoveProbabilitiesAndWeights(self.referenceclass)
        return {'weigther': weighter}

    def convertFromSourceFile(self, filename, weighterobjects, istraining):

        # Function to produce the numpy training arrays from root files

        from DeepJetCore.Weighter import Weighter
        from DeepJetCore.stopwatch import stopwatch
        sw = stopwatch()
        swall = stopwatch()
        if not istraining:
            self.remove = False

        print('reading '+filename)
        import ROOT
        from root_numpy import tree2array, root2array
        fileTimeOut(filename, 120)  # give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("tree")
        self.nsamples = tree.GetEntries()

        # user code, example works with the example 2D images in root format generated by make_example_data
        from DeepJetCore.preprocessing import MeanNormZeroPad, MeanNormZeroPadParticles

        print('padding '+filename)

        x_global = MeanNormZeroPad(filename, None,  # 2nd argument None: should mean no Normalisation (makes sense, because how would the first file know of the normalisation of the last one?)
                                   [self.global_branches],
                                   [1], self.nsamples)

        x_pfCand_neutral = MeanNormZeroPadParticles(filename, None,
                                                    self.pfCand_neutral_branches,
                                                    self.npfCand_neutral, self.nsamples)
        try:
            print("this is for v3 unbalanced")
            print("x global shape is {}".format(x_global.shape))
            print("x pfCand neutral shape is {}".format(x_pfCand_neutral.shape))    
            # for the first file and for the second file: (shapes)
            # x_globa: (185, 25)    x_pfCand_neutral: (185, 10, 7) 
            # x_globa: (234, 25)    x_pfCand_neutral: (234, 10, 7)
            # Explaination:
            # in the first file we have 185 leptons and 25 global features for them
            # we make an array with max 10 neutral cand (fill the rest with zeros if 
            # we have less than 10 neutral candidates) and for the neutral candidates
            # we have 7 features.

        except:
            pass
        #print("x_global is (we called meannormzeropad) {}".format(x_global))
        #print("x_pfCand_neutral is {}".format(x_pfCand_neutral))
        x_pfCand_charged = MeanNormZeroPadParticles(filename, None,
                                                    self.pfCand_charged_branches,
                                                    self.npfCand_charged, self.nsamples)

        x_pfCand_photon = MeanNormZeroPadParticles(filename, None,
                                                   self.pfCand_photon_branches,
                                                   self.npfCand_photon, self.nsamples)

        x_pfCand_electron = MeanNormZeroPadParticles(filename, None,
                                                     self.pfCand_electron_branches,
                                                     self.npfCand_electron, self.nsamples)

        x_pfCand_muon = MeanNormZeroPadParticles(filename, None,
                                                 self.pfCand_muon_branches,
                                                 self.npfCand_muon, self.nsamples)

        x_pfCand_SV = MeanNormZeroPadParticles(filename, None,
                                               self.SV_branches,
                                               self.nSV, self.nsamples)


        import uproot3 as uproot
        urfile = uproot.open(filename)["tree"]
        # WRITE AS FUNCTION OF THE TRUTH BRANCHES
        truth = np.concatenate([np.expand_dims(urfile.array("lep_isPromptId_Training"), axis=1),
                                np.expand_dims(urfile.array("lep_isNonPromptId_Training"), axis=1),
                                np.expand_dims(urfile.array("lep_isFakeId_Training"), axis=1),
                                np.expand_dims(urfile.array("lep_isFromSUSY_Training"), axis=1),
                                np.expand_dims(urfile.array("lep_isFromSUSYHF_Training"), axis=1),], axis=1)

        # print("the truth shape is {}".format(truth.shape))
        # the truth shape is (185, 5) in the first file and
        # (234, 5) in the second file

        # important, float32 and C-type!
        truth = truth.astype(dtype='float32', order='C')

        x_global = x_global.astype(dtype='float32', order='C')
        x_pfCand_neutral = x_pfCand_neutral.astype(dtype='float32', order='C')
        x_pfCand_charged = x_pfCand_charged.astype(dtype='float32', order='C')
        x_pfCand_photon = x_pfCand_photon.astype(dtype='float32', order='C')
        x_pfCand_electron = x_pfCand_electron.astype(
            dtype='float32', order='C')
        x_pfCand_muon = x_pfCand_muon.astype(dtype='float32', order='C')
        x_pfCand_SV = x_pfCand_SV.astype(dtype='float32', order='C')

        if self.remove:
            b = [self.weightbranchX, self.weightbranchY]
            b.extend(self.truth_branches)
            b.extend(self.undefTruth)
            fileTimeOut(filename, 120)
            for_remove = root2array( # gives a structured np array
                filename,
                treename="tree",
                stop=None,
                branches=b
            )
            notremoves = weighterobjects['weigther'].createNotRemoveIndices(
                for_remove)
            # undef=for_remove['isUndefined']
            # notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')
            # if counter_all == 0:
            #    notremoves = list(np.ones(np.shape(notremoves)))

        if self.remove:
            # print('remove')
            print("notremoves", notremoves, "<- notremoves")
            # the lenght of this 'array' is 185 for the first file
            # so we decide whether we take the lep or not
#           notremoves [0. 1. 0. 1. 0. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 1. 1. 1. 0. 1. 0. 0. 1.
#            1. 1. 1. 1. 1. 0. 1. 1. 0. 1. 1. 0. 1. 0. 0. 1. 1. 1. 1. 1. 1. 0. 1. 0.
#            1. 0. 1. 0. 1. 0. 0. 1. 0. 0. 1. 0. 1. 1. 1. 0. 0. 1. 0. 1. 0. 0. 1. 1.
#            0. 1. 0. 0. 0. 1. 1. 1. 1. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0.
#            1. 0. 1. 1. 1. 1. 0. 1. 0. 1. 1. 1. 1. 1. 0. 1. 1. 1. 0. 1. 0. 1. 1. 1.
#            0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 1. 0. 1. 1. 0. 1. 1. 1. 0. 0. 1. 1. 0. 1.
#            0. 1. 1. 1. 1. 0. 1. 1. 1. 0. 0. 1. 1. 1. 1. 1. 1. 0. 0. 0. 1. 1. 0. 1.
#            0. 1. 0. 1. 1. 0. 1. 1. 0. 1. 0. 0. 0. 0. 0. 1. 1.] <- notremoves
#           reduced content to  55 %
#           



            x_global = x_global[notremoves > 0]
            x_pfCand_neutral = x_pfCand_neutral[notremoves > 0]
            x_pfCand_charged = x_pfCand_charged[notremoves > 0]
            x_pfCand_photon = x_pfCand_photon[notremoves > 0]
            x_pfCand_electron = x_pfCand_electron[notremoves > 0]
            x_pfCand_muon = x_pfCand_muon[notremoves > 0]
            x_pfCand_SV = x_pfCand_SV[notremoves > 0]
            truth = truth[notremoves > 0]

        newnsamp = x_global.shape[0]
        print('reduced content to ', int(
            float(newnsamp)/float(self.nsamples)*100), '%')
        # print(x_global)
        # print(x_pfCand_neutral)
        # print(x_pfCand_charged)
        # print(x_pfCand_photon)
        # print(x_pfCand_electron)
        # print(x_pfCand_muon)
        # print(x_pfCand_SV)

        print('remove nans')
        x_global = np.where(np.isfinite(x_global), x_global, 0)
        x_pfCand_neutral = np.where(np.isfinite(
            x_pfCand_neutral), x_pfCand_neutral, 0)
        x_pfCand_charged = np.where(np.isfinite(
            x_pfCand_charged), x_pfCand_charged, 0)
        x_pfCand_photon = np.where(np.isfinite(
            x_pfCand_photon), x_pfCand_photon, 0)
        x_pfCand_electron = np.where(np.isfinite(
            x_pfCand_electron), x_pfCand_electron, 0)
        x_pfCand_muon = np.where(np.isfinite(x_pfCand_muon), x_pfCand_muon, 0)
        x_pfCand_SV = np.where(np.isfinite(x_pfCand_SV), x_pfCand_SV, 0)

        return [x_global, x_pfCand_neutral, x_pfCand_charged, x_pfCand_photon, x_pfCand_electron, x_pfCand_muon, x_pfCand_SV], [truth], []

    # defines how to write out the prediction
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        print("pred started")
        # import os
        # predicted will be a list

        if self.firstfilepred:
            self.firstfilepred = False
            print("Investigating predicted:")
            print("len = {}".format(len(predicted)))
            for i in range(len(predicted)):
                print("shape and type of ith element: {}, {}".format(predicted[i].shape, type(predicted[i])))
                # print("np dtype: {}".format(predicted[i].dtype)) not a structured array
            print("Investigating features:")
            print("len = {}".format(len(features)))
            for i in range(len(features)):
                print("shape and type of ith element: {}, {}".format(features[i].shape, type(features[i])))
                # print("dtype of features: {}".format(features[i].dtype)) not a structured array
            print("Investigating truth:")
            print("len = {}".format(len(truth)))
            for i in range(len(truth)):
                print("shape and type of ith element: {}, {}".format(truth[i].shape, type(truth[i])))
                # print("dtype: {}".format(truth[i].dtype)) -> float so its not a structured array
            print("Investigating weights:")
            print("len = {}".format(len(weights)))
            for i in range(len(weights)):
                print("shape and type of ith element: {}, {}".format(weights[i].shape, type(weights[i])))
            
            # np.set_printoptions(threshold=np.inf)
            nleps = len(predicted[0][:,0])
            printmaxleps = 50
            fname, fext = os.path.splitext(outfilename)
            with open("{}_prediction.txt".format(fname), 'w') as f:
                f.write("{:-^60s}|{:-^60s}\n".format("Prediction", "Truth"))
                # print("{:-^60s}|{:-^60s}".format("Prediction", "Truth"))
                tempstring = ""
                for label in self.truth_branches:
                    tempstring+="{:^12s}".format( (label.replace("lep_is", "")).replace("_Training", ""))
                f.write("{0:^60s}|{0:^60s}\n".format(tempstring))
                f.write("{:-^121}\n".format(""))
                for i in range(nleps if nleps<printmaxleps else printmaxleps):
                    pred_str = "{0[0]:^10.2f}, {0[1]:^10.2f}, {0[2]:^10.2f}, {0[3]:^10.2f}, {0[4]:^10.2f}".format(predicted[0][i,:])
                    truth_str = "{0[0]:^10.0f}, {0[1]:^10.0f}, {0[2]:^10.0f}, {0[3]:^10.0f}, {0[4]:^10.0f}".format(truth[0][i,:])
                    f.write("{:^60s}|{:^60s}\n".format(pred_str, truth_str))
                    # print("{:^60s}|{:^60s}".format(pred_str, truth_str))

            # raise NotImplemented("stop you here")
            # Investigating predicted:
            # len = 1
            # shape and type of ith element: (254, 5), <class 'numpy.ndarray'>
            # Investigating features:
            # len = 7
            # shape and type of ith element: (254, 25), <class 'numpy.ndarray'>
            # shape and type of ith element: (254, 10, 7), <class 'numpy.ndarray'>
            # shape and type of ith element: (254, 80, 18), <class 'numpy.ndarray'>
            # shape and type of ith element: (254, 50, 7), <class 'numpy.ndarray'>
            # shape and type of ith element: (254, 4, 18), <class 'numpy.ndarray'>
            # shape and type of ith element: (254, 6, 18), <class 'numpy.ndarray'>
            # shape and type of ith element: (254, 10, 16), <class 'numpy.ndarray'>
            # Investigating truth:
            # len = 1
            # shape and type of ith element: (254, 5), <class 'numpy.ndarray'>
            # Investigating weights:
            # len = 0




        # print("predicted: {}".format(predicted))
        # print("features: {}".format(features))
        # print("truth: {}".format(truth))
        # print("weights: {}".format(weights))
        # raise NotImplemented("check the code before running")
        from root_numpy import array2root
        # from numpy source code:
        # Record arrays allow us to access fields as properties: -> out.x, when before out['x'] (structured arrays)
        namesstring = 'prob_isPrompt, prob_isNonPrompt, prob_isFake, prob_isFromSUSY, prob_isFromSUSYHF,'
        for label in self.truth_branches:
            namesstring += label + ', '
        namesstring += 'lep_pt, lep_dxy'
        out = np.core.records.fromarrays(np.vstack((predicted[0].transpose(), truth[0].transpose(), features[0][:, [0,6]].transpose())), names=namesstring)
        print("Information about out array:")
        print("dtype is {}".format(out.dtype))
        print("shape is {}".format(out.shape))
        array2root(out, outfilename, 'tree')


