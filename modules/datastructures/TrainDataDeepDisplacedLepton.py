from DeepJetCore.TrainData import TrainData
from DeepJetCore.TrainData import fileTimeOut as tdfto
import numpy as np
import os
import DeepJetCore.preprocessing
# set tree name to use
DeepJetCore.preprocessing.setTreeName('tree')


def fileTimeOut(fileName, timeOut):
    tdfto(fileName, timeOut)


class TrainDataDeepDisplacedLepton_dxy(TrainData):
    '''
    Base class for DeepLepton Training Data.
    '''

    def __init__(self):
        TrainData.__init__(self)

        self.description = "DeepLepton training datastructure"
        # Define the four truth labels, should be consistent with
        # Network architecture, i.e. the output layer.
        self.truth_branches = ['lep_isPromptId_Training',
                               'lep_isNonPromptId_Training',
                               'lep_isFakeId_Training',
                               'lep_isFromSUSYandHF_Training']
                              #  'lep_isFromSUSY_Training',
                              #  'lep_isFromSUSYHF_Training',]  
        self.undefTruth = []
        # The Weighter can "weight" two lepton features.
        # Here we only "weight" the dxy feature by making only one bin in eta
        # "weight" means to throw away leptons from the non reference class s.t.
        # weightbranchX/Y gets similarly distributed as the reference class
        self.weightbranchX = 'lep_eta' 
        self.weightbranchY = 'lep_dxy'
        # enables or disables (False) Weighter
        self.remove = True
        # Define the reference class
        self.referenceclass = 'lep_isFromSUSYandHF_Training' 
        # setting DeepLepton specific defaults
        self.treename = "tree"

        # make one huge bin for eta
        self.weight_binX = np.array([-30, 30])
        # make the extension large enough to cover the whole feature range of the 
        # leptons in the non-reference class
        self.weight_binY = np.linspace(start=-50, stop=50, num=50, endpoint=True, dtype=float)
        
        # if training not flat in pt, eta one could remove the
        # feature from the training as well
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
        # Calculates the weights needed for flattening the pt/eta spectrum
        # by initializing a Weighter instance

        from DeepJetCore.Weighter import Weighter
        weighter = Weighter()
        weighter.undefTruth = self.undefTruth
        branches = [self.weightbranchX, self.weightbranchY]
        branches.extend(self.truth_branches)
        
        # if one uses the Weighter
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

        x_global = MeanNormZeroPad(filename, None,  # 2nd argument None: means no normalisation 
                                   [self.global_branches],
                                   [1], self.nsamples)

        x_pfCand_neutral = MeanNormZeroPadParticles(filename, None,
                                                    self.pfCand_neutral_branches,
                                                    self.npfCand_neutral, self.nsamples)

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

        mytruth = []
        for arr in self.truth_branches:
            mytruth.append(np.expand_dims(urfile.array(arr), axis=1))
        truth = np.concatenate(mytruth, axis = 1)

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
            for_remove = root2array( # returns a structured np array
                filename,
                treename="tree",
                stop=None,
                branches=b
            )
            notremoves = weighterobjects['weigther'].createNotRemoveIndices(
                for_remove)
            print('took ', sw.getAndReset(), ' to create remove indices')

        if self.remove:

            x_global = x_global[notremoves > 0]
            x_pfCand_neutral = x_pfCand_neutral[notremoves > 0]
            x_pfCand_charged = x_pfCand_charged[notremoves > 0]
            x_pfCand_photon = x_pfCand_photon[notremoves > 0]
            x_pfCand_electron = x_pfCand_electron[notremoves > 0]
            x_pfCand_muon = x_pfCand_muon[notremoves > 0]
            x_pfCand_SV = x_pfCand_SV[notremoves > 0]
            truth = truth[notremoves > 0]

        newnsamp = x_global.shape[0]
        print('Weighter reduced content to ', int(
            float(newnsamp)/float(self.nsamples)*100), '%')

        print('removing nans')
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
        # predicted is a list
        print("Prediction started")
        from root_numpy import array2root

        namesstring = 'prob_isPrompt, prob_isNonPrompt, prob_isFake, prob_lep_isFromSUSYandHF,' #prob_isFromSUSY, prob_isFromSUSYHF,'
        for label in self.truth_branches:
            namesstring += label + ', '
        features_string = ', '.join(self.global_branches)
        namesstring += features_string
        out = np.core.records.fromarrays(np.vstack((predicted[0].transpose(), truth[0].transpose(), features[0][:, :].transpose())), names=namesstring)

        # if one predicts on a DataCollection one has to change 
        # the file extension to .root
        if not outfilename.endswith(".root"):
            print("Predicted from a DataCollection make root files")
            print("Note that outfiles files extensions must be adapted...")
            filename, _ = os.path.splitext(outfilename)
            outfilename = filename+".root"
        print("making {}".format(outfilename.split('/')[-1]))   
        array2root(out, outfilename, 'tree')


