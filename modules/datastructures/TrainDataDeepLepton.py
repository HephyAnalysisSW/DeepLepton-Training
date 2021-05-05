from DeepJetCore.TrainData import TrainData
from DeepJetCore.TrainData import fileTimeOut as tdfto
import numpy as np

#set tree name to use
import DeepJetCore.preprocessing
DeepJetCore.preprocessing.setTreeName('tree')

def fileTimeOut(fileName, timeOut):
    tdfto(fileName, timeOut)

class TrainDataDeepLepton(TrainData):
    '''
    Base class for DeepLepton Training Data.
    '''
    
    def __init__(self):
        TrainData.__init__(self)

        self.description      = "DeepLepton training datastructure"
        self.truth_branches   = ['lep_isPromptId_Training','lep_isNonPromptId_Training','lep_isFakeId_Training']
        self.undefTruth       = []
        self.weightbranchX    = 'lep_pt'
        self.weightbranchY    = 'lep_eta'
        self.remove           = True
        self.referenceclass   = 'lep_isNonPromptId_Training' 
        #setting DeepLepton specific defaults
        self.treename         = "tree"
        #self.undefTruth=['isUndefined']
        #self.red_classes      = ['cat_P', 'cat_NP', 'cat_F']
        #self.reduce_truth     = ['lep_isPromptId_Training', 'lep_isNonPromptId_Training', 'lep_isFakeId_Training']
        #self.class_weights    = [1.00, 1.00, 1.00]

        self.weight_binX = np.array([
                5,7.5,10,12.5,15,17.5,20,25,30,35,40,45,50,60,75,100,
                125,150,175,200,250,300,400,500,
                600,2000],dtype=float)
        #self.weight_binX = np.geomspace(3.5, 2000, 30)
        
        self.weight_binY = np.array(
            [-2.5,-2.,-1.5,-1.,-0.5,0.5,1,1.5,2.,2.5],
            dtype=float
            )

        self.global_branches = [
            'lep_pt', 'lep_eta', 'lep_phi',
            'lep_mediumId',
            'lep_miniPFRelIso_all',# 'lep_pfRelIso03_all',
            'lep_sip3d', 'lep_dxy', 'lep_dz',
            'lep_charge',
            'lep_dxyErr', 'lep_dzErr', 'lep_ip3d',
            'lep_jetPtRelv2', 'lep_jetRelIso',
            'lep_miniPFRelIso_chg', 'lep_mvaLowPt', 'lep_nStations', 'lep_nTrackerLayers', 'lep_pfRelIso03_all', 'lep_pfRelIso03_chg', 'lep_pfRelIso04_all', 'lep_ptErr',
            'lep_segmentComp', 'lep_tkRelIso', 'lep_tunepRelPt',
            ]

        self.pfCand_neutral_branches = ['pfCand_neutral_eta', 'pfCand_neutral_phi', 'pfCand_neutral_pt', 'pfCand_neutral_puppiWeight', 'pfCand_neutral_puppiWeightNoLep', 'pfCand_neutral_ptRel', 'pfCand_neutral_deltaR',]
        self.npfCand_neutral         = 10

        self.pfCand_charged_branches = ['pfCand_charged_d0', 'pfCand_charged_d0Err', 'pfCand_charged_dz', 'pfCand_charged_dzErr', 'pfCand_charged_eta', 'pfCand_charged_mass', 'pfCand_charged_phi', 'pfCand_charged_pt', 'pfCand_charged_puppiWeight', 'pfCand_charged_puppiWeightNoLep', 'pfCand_charged_trkChi2', 'pfCand_charged_vtxChi2', 'pfCand_charged_charge', 'pfCand_charged_lostInnerHits', 'pfCand_charged_pvAssocQuality', 'pfCand_charged_trkQuality', 'pfCand_charged_ptRel', 'pfCand_charged_deltaR',]
        self.npfCand_charged         = 80

        self.pfCand_photon_branches  = ['pfCand_photon_eta', 'pfCand_photon_phi', 'pfCand_photon_pt', 'pfCand_photon_puppiWeight', 'pfCand_photon_puppiWeightNoLep', 'pfCand_photon_ptRel', 'pfCand_photon_deltaR',]
        self.npfCand_photon          = 50

        self.pfCand_electron_branches = ['pfCand_electron_d0', 'pfCand_electron_d0Err', 'pfCand_electron_dz', 'pfCand_electron_dzErr', 'pfCand_electron_eta', 'pfCand_electron_mass', 'pfCand_electron_phi', 'pfCand_electron_pt', 'pfCand_electron_puppiWeight', 'pfCand_electron_puppiWeightNoLep', 'pfCand_electron_trkChi2', 'pfCand_electron_vtxChi2', 'pfCand_electron_charge', 'pfCand_electron_lostInnerHits', 'pfCand_electron_pvAssocQuality', 'pfCand_electron_trkQuality', 'pfCand_electron_ptRel', 'pfCand_electron_deltaR',]
        self.npfCand_electron         = 4

        self.pfCand_muon_branches = ['pfCand_muon_d0', 'pfCand_muon_d0Err', 'pfCand_muon_dz', 'pfCand_muon_dzErr', 'pfCand_muon_eta', 'pfCand_muon_mass', 'pfCand_muon_phi', 'pfCand_muon_pt', 'pfCand_muon_puppiWeight', 'pfCand_muon_puppiWeightNoLep', 'pfCand_muon_trkChi2', 'pfCand_muon_vtxChi2', 'pfCand_muon_charge', 'pfCand_muon_lostInnerHits', 'pfCand_muon_pvAssocQuality', 'pfCand_muon_trkQuality', 'pfCand_muon_ptRel', 'pfCand_muon_deltaR']
        self.npfCand_muon         = 6

        self.SV_branches = ['SV_dlen', 'SV_dlenSig', 'SV_dxy', 'SV_dxySig', 'SV_pAngle', 'SV_chi2', 'SV_eta', 'SV_mass', 'SV_ndof', 'SV_phi', 'SV_pt', 'SV_x', 'SV_y', 'SV_z', 'SV_ptRel', 'SV_deltaR',]
        self.nSV         = 10

    def createWeighterObjects(self, allsourcefiles):
        # 
        # Calculates the weights needed for flattening the pt/eta spectrum
        
        from DeepJetCore.Weighter import Weighter
        weighter            = Weighter()
        weighter.undefTruth = self.undefTruth
        branches            = [self.weightbranchX,self.weightbranchY]
        branches.extend(self.truth_branches)

        if self.remove:
            weighter.setBinningAndClasses(
                [self.weight_binX,  self.weight_binY],
                self.weightbranchX, self.weightbranchY,
                self.truth_branches,
                method = self.referenceclass
            )
        
        counter=0
        import ROOT
        from root_numpy import tree2array, root2array
        if self.remove:
            for fname in allsourcefiles:
                fileTimeOut(fname, 120)
                nparray = root2array(
                    fname,
                    treename = "tree",
                    stop = None,
                    branches = branches
                )
                norm_hist = True
                if self.referenceclass == 'flatten':
                    norm_hist = False
                weighter.addDistributions(nparray, norm_h = norm_hist)
                #del nparray
                counter=counter+1
            weighter.createRemoveProbabilitiesAndWeights(self.referenceclass)
        return {'weigther':weighter}
    
    def convertFromSourceFile(self, filename, weighterobjects, istraining):

        # Function to produce the numpy training arrays from root files

        from DeepJetCore.Weighter import Weighter
        from DeepJetCore.stopwatch import stopwatch
        sw    = stopwatch()
        swall = stopwatch()
        if not istraining:
            self.remove = False

        #def reduceTruth(uproot_arrays):
        #    #import numpy as np
        #    prompt    = uproot_arrays[b'lep_isPromptId_Training']
        #    nonPrompt = uproot_arrays[b'lep_isNonPromptId_Training']
        #    fake      = uproot_arrays[b'lep_isFakeId_Training']
        #    print (prompt, nonPrompt, fake)
        #    return np.vstack((prompt, nonPrompt, fake)).transpose()
        #    #return np.concatenate( [ prompt, nonPrompt, fake] )
        
        print('reading '+filename)
        import ROOT
        from root_numpy import tree2array, root2array
        fileTimeOut(filename,120) #give eos a minute to recover
        rfile = ROOT.TFile(filename)
        tree = rfile.Get("tree")
        self.nsamples = tree.GetEntries()
        
        # user code, example works with the example 2D images in root format generated by make_example_data
        from DeepJetCore.preprocessing import MeanNormZeroPad,MeanNormZeroPadParticles
        
        print('padding '+filename)
        # globalvars
        #globalarr = tree2array(tree, self.global_branches)
        #x_global = np.array([list(x) for x in globalarr])
                
        def padding(arr, nbranches, nCands, nlist):
            zeros = [0. for i in range(nbranches)]
            padded = []
            print(len(nlist))
            zeroarr = np.zeros((len(nlist), nCands, nbranches))
            cntr = 0
            for a, N in zip(arr, nlist):
                #pad = []
                aa = [[xx for xx in x] for x in a] #transform list of arrays into list of lists
                n = min(N, nCands) # to not get index out of bounds
                for i in range(n): # add numbers
                    zeroarr[cntr, i] = np.array([row[i] for row in aa], dtype = np.float32)
                if cntr % 10000 == 0:
                    print(cntr, ' of ', len(nlist))
                cntr += 1
            return zeroarr

        
        #neutralarr = tree2array(tree, self.pfCand_neutral_branches)
        #neutrallist = [list(x) for x in neutralarr]
        #x_pfCand_neutral2 = padding(neutrallist, len(self.pfCand_neutral_branches), self.npfCand_neutral, tree2array(tree, 'npfCand_neutral'))
        #print('n done')
        #chargedarr = tree2array(tree, self.pfCand_charged_branches)
        #chargedlist = [list(x) for x in chargedarr]
        #x_pfCand_charged2 = padding(chargedlist, len(self.pfCand_charged_branches), self.npfCand_charged, tree2array(tree, 'npfCand_charged'))
        #print('c done')
        #photonarr = tree2array(tree, self.pfCand_photon_branches)
        #photonlist = [list(x) for x in photonarr]
        #x_pfCand_photon2 = padding(photonlist, len(self.pfCand_photon_branches), self.npfCand_photon, tree2array(tree, 'npfCand_photon'))
        #print('p done')
        #electronarr = tree2array(tree, self.pfCand_electron_branches)
        #electronlist = [list(x) for x in electronarr]
        #x_pfCand_electron2 = padding(electronlist, len(self.pfCand_electron_branches), self.npfCand_electron, tree2array(tree, 'npfCand_electron'))
        #print('e done')
        #muonarr = tree2array(tree, self.pfCand_muon_branches)
        #muonlist = [list(x) for x in muonarr]
        #x_pfCand_muon2 = padding(muonlist, len(self.pfCand_muon_branches), self.npfCand_muon, tree2array(tree, 'npfCand_muon'))
        #print('m done')
        #SVarr = tree2array(tree, self.SV_branches)
        #SVlist = [list(x) for x in SVarr]
        #x_pfCand_SV2 = padding(SVlist, len(self.SV_branches), self.nSV, tree2array(tree, 'nSV'))
        #print('s done')
        #print('done padding '+ filename)

        x_global = MeanNormZeroPad(filename,None, # 2nd argument None: should mean no Normalisation (makes sense, because how would the first file know of the normalisation of the last one?)
                                   [self.global_branches],
                                   [1],self.nsamples)

        x_pfCand_neutral = MeanNormZeroPadParticles(filename,None,
                                   self.pfCand_neutral_branches,
                                   self.npfCand_neutral,self.nsamples)
        x_pfCand_charged = MeanNormZeroPadParticles(filename,None,
                                   self.pfCand_charged_branches,
                                   self.npfCand_charged,self.nsamples)

        x_pfCand_photon = MeanNormZeroPadParticles(filename,None,
                                   self.pfCand_photon_branches,
                                   self.npfCand_photon,self.nsamples)
        
        x_pfCand_electron = MeanNormZeroPadParticles(filename,None,
                                   self.pfCand_electron_branches,
                                   self.npfCand_electron,self.nsamples)
        
        x_pfCand_muon = MeanNormZeroPadParticles(filename,None,
                                   self.pfCand_muon_branches,
                                   self.npfCand_muon,self.nsamples)

        x_pfCand_SV = MeanNormZeroPadParticles(filename,None,
                                   self.SV_branches,
                                   self.nSV,self.nsamples)
        
        #import uproot3 as uproot
        #urfile       = uproot.open(filename)["tree"]
        #truth_arrays = urfile.arrays(self.truth_branches)
        #truth        = reduceTruth(truth_arrays)
        #truth        = truth.astype(dtype='float32', order='C') #important, float32 and C-type!

        import uproot3 as uproot
        urfile = uproot.open(filename)["tree"]
        truth = np.concatenate([np.expand_dims(urfile.array("lep_isPromptId_Training"), axis=1) ,
                                np.expand_dims(urfile.array("lep_isNonPromptId_Training"), axis=1),
                                np.expand_dims(urfile.array("lep_isFakeId_Training"), axis=1)],axis=1)
        truth = truth.astype(dtype='float32', order='C') #important, float32 and C-type!

        x_global            = x_global.astype(dtype='float32', order='C')
        x_pfCand_neutral    = x_pfCand_neutral.astype(dtype='float32', order='C')
        x_pfCand_charged    = x_pfCand_charged.astype(dtype='float32', order='C')
        x_pfCand_photon     = x_pfCand_photon.astype(dtype='float32', order='C')
        x_pfCand_electron   = x_pfCand_electron.astype(dtype='float32', order='C')
        x_pfCand_muon       = x_pfCand_muon.astype(dtype='float32', order='C')
        x_pfCand_SV         = x_pfCand_SV.astype(dtype='float32', order='C')

        if self.remove:
            b = [self.weightbranchX,self.weightbranchY]
            b.extend(self.truth_branches)
            b.extend(self.undefTruth)
            fileTimeOut(filename, 120)
            for_remove = root2array(
                filename,
                treename = "tree",
                stop = None,
                branches = b
            )
            notremoves=weighterobjects['weigther'].createNotRemoveIndices(for_remove)
            #undef=for_remove['isUndefined']
            #notremoves-=undef
            print('took ', sw.getAndReset(), ' to create remove indices')
            #if counter_all == 0:
            #    notremoves = list(np.ones(np.shape(notremoves)))
                
        if self.remove:
            #print('remove')
            print ("notremoves", notremoves, "<- notremoves")
            x_global            =   x_global[notremoves > 0]
            x_pfCand_neutral    =   x_pfCand_neutral[notremoves > 0]
            x_pfCand_charged    =   x_pfCand_charged[notremoves > 0]
            x_pfCand_photon     =   x_pfCand_photon[notremoves > 0]
            x_pfCand_electron   =   x_pfCand_electron[notremoves > 0]
            x_pfCand_muon       =   x_pfCand_muon[notremoves > 0]
            x_pfCand_SV         =   x_pfCand_SV[notremoves > 0]
            truth               =   truth[notremoves > 0]

        newnsamp=x_global.shape[0]
        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
        #print(x_global)
        #print(x_pfCand_neutral)
        #print(x_pfCand_charged)
        #print(x_pfCand_photon)
        #print(x_pfCand_electron)
        #print(x_pfCand_muon)
        #print(x_pfCand_SV)
        
        print('remove nans')
        x_global          = np.where(np.isfinite(x_global) , x_global, 0)
        x_pfCand_neutral  = np.where(np.isfinite(x_pfCand_neutral), x_pfCand_neutral, 0)
        x_pfCand_charged  = np.where(np.isfinite(x_pfCand_charged), x_pfCand_charged, 0)
        x_pfCand_photon   = np.where(np.isfinite(x_pfCand_photon), x_pfCand_photon, 0)
        x_pfCand_electron = np.where(np.isfinite(x_pfCand_electron), x_pfCand_electron, 0)
        x_pfCand_muon     = np.where(np.isfinite(x_pfCand_muon), x_pfCand_muon, 0)
        x_pfCand_SV       = np.where(np.isfinite(x_pfCand_SV), x_pfCand_SV, 0)

        return [x_global, x_pfCand_neutral, x_pfCand_charged, x_pfCand_photon, x_pfCand_electron, x_pfCand_muon, x_pfCand_SV], [truth], []
    
    ## defines how to write out the prediction
    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        # predicted will be a list
        
        from root_numpy import array2root
        out = np.core.records.fromarrays(np.vstack( (predicted[0].transpose(),truth[0].transpose(), features[0][:,0:2].transpose() ) ),
                                         names='prob_isPrompt, prob_isNonPrompt, prob_isFake, lep_pt, lep_eta')
        array2root(out, outfilename, 'tree')


#    def readTreeFromRootToTuple(self, filenames, limit=None, branches=None):
#        '''
#        To be used to get the initial tupel for further processing in inherting classes
#        Makes sure the number of entries is properly set
#        
#        can also read a list of files (e.g. to produce weights/removes from larger statistics
#        (not fully tested, yet)
#        '''
#
#        if branches is None or len(branches) == 0:
#            return np.array([],dtype='float32')
#
#        #print(branches)
#        #remove duplicates
#        usebranches=list(set(branches))
#        tmpbb=[]
#        for b in usebranches:
#            if len(b):
#                tmpbb.append(b)
#        usebranches=tmpbb
#
#        import ROOT
#        from root_numpy import tree2array, root2array
#        if isinstance(filenames, list):
#            for f in filenames:
#                fileTimeOut(f,120)
#            print('add files')
#            nparray = root2array(
#                filenames,
#                treename = "tree",
#                stop = limit,
#                branches = usebranches
#                )
#            print('done add files')
#            return nparray
#            print('add files')
#        else:
#            fileTimeOut(filenames,120) #give eos a minute to recover
#            rfile = ROOT.TFile(filenames)
#            tree = rfile.Get(self.treename)
#            if not self.nsamples:
#                self.nsamples=tree.GetEntries()
#            nparray = tree2array(tree, stop=limit, branches=usebranches)
#            return nparray
#
#    def createWeighterObjects(self, allsourcefiles):
#        # 
#        # Calculates the weights needed for flattening the pt/eta spectrum
#
#        from DeepJetCore.Weighter import Weighter
#        weighter = Weighter()
#        weighter.undefTruth = self.undefTruth
#        weighter.class_weights = self.class_weights
#        branches = [self.weightbranchX,self.weightbranchY]
#        branches.extend(self.truth_branches)
#
#        if self.remove:
#            weighter.setBinningAndClasses(
#                [self.weight_binX,self.weight_binY],
#                self.weightbranchX,self.weightbranchY,
#                self.truth_branches, self.red_classes,
#                self.truth_red_fusion, method = self.referenceclass
#            )
#
#        counter=0
#        import ROOT
#        from root_numpy import tree2array, root2array
#        if self.remove:
#            for fname in allsourcefiles:
#                fileTimeOut(fname, 120)
#                nparray = root2array(
#                    fname,
#                    treename = "tree",
#                    stop = None,
#                    branches = branches
#                )
#                norm_hist = True
#                if self.referenceclass == 'flatten':
#                    norm_hist = False
#                weighter.addDistributions(nparray, norm_h = norm_hist)
#                #del nparray
#                counter=counter+1
#            weighter.createRemoveProbabilitiesAndWeights(self.referenceclass)
#
#        print("calculate means")
#        from DeepJetCore.preprocessing import meanNormProd
#        nparray = self.readTreeFromRootToTuple(allsourcefiles,branches=self.global_branches+self.pfCand_neutral_branches+self.pfCand_charged_branches+self.pfCand_photon_branches+self.pfCand_electron_branches+self.pfCand_muon_branches+self.SV_branches, limit=500000)
#        for a in (self.global_branches+self.pfCand_neutral_branches+self.pfCand_charged_branches+self.pfCand_photon_branches+self.pfCand_electron_branches+self.pfCand_muon_branches+self.SV_branches):
#            for b in range(len(nparray[a])):
#                nparray[a][b] = np.where(np.logical_and(np.isfinite(nparray[a][b]),np.abs(nparray[a][b]) < 100000.0), nparray[a][b], 0)
#        means = np.array([],dtype='float32')
#        if len(nparray):
#            means = meanNormProd(nparray)
#        return {'weigther':weighter,'means':means}
#
#    def convertFromSourceFile(self, filename, weighterobjects, istraining):
#
#        # Function to produce the numpy training arrays from root files
#
#        from DeepJetCore.Weighter import Weighter
#        from DeepJetCore.stopwatch import stopwatch
#        sw=stopwatch()
#        swall=stopwatch()
#        if not istraining:
#            self.remove = False
#
#        def reduceTruth(uproot_arrays):
#            p   = uproot_arrays[b'lep_isPromptId_Training']
#            np  = uproot_arrays[b'lep_isNonPromptId_Training']
#            f   = uproot_arrays[b'lep_isFakeId_Training']
#
#            return np.vstack((p, np, f)).transpose()
#
#        print('reading '+filename)
#
#        import ROOT
#        from root_numpy import tree2array, root2array
#        fileTimeOut(filename,120) #give eos a minute to recover
#        rfile = ROOT.TFile(filename)
#        tree = rfile.Get("tree")
#        self.nsamples = tree.GetEntries()
#        # user code, example works with the example 2D images in root format generated by make_example_data
#        from DeepJetCore.preprocessing import MeanNormZeroPad,MeanNormZeroPadParticles
#        x_global = MeanNormZeroPad(filename,weighterobjects['means'],
#                                   [self.global_branches, self.pfCand_neutral_branches, self.pfCand_charged_branches, self.pfCand_photon_branches, self.pfCand_electron_branches, self.pfCand_muon_branches, self.SV_branches],
#                                   [1,self.npfCand_neutral,self.npfCand_charged,self.npfCand_photon,self.npfCand_electron, self.npfCand_muon, self.nSV],
#)
#
#        import uproot3 as uproot
#        urfile = uproot.open(filename)["tree"]
#        truth_arrays = urfile.arrays(self.truth_branches)
#        truth = reduceTruth(truth_arrays)
#        truth = truth.astype(dtype='float32', order='C') #important, float32 and C-type!
#
#        x_global = x_global.astype(dtype='float32', order='C')
#
#        if self.remove:
#            b = [self.weightbranchX,self.weightbranchY]
#            b.extend(self.truth_branches)
#            b.extend(self.undefTruth)
#            fileTimeOut(filename, 120)
#            for_remove = root2array(
#                filename,
#                treename = "tree",
#                stop = None,
#                branches = b
#            )
#            notremoves=weighterobjects['weigther'].createNotRemoveIndices(for_remove)
#            #undef=for_remove['isUndefined']
#            #notremoves-=undef
#            print('took ', sw.getAndReset(), ' to create remove indices')
#
#        if self.remove:
#            print('remove')
#            x_global=x_global[notremoves > 0]
#            truth=truth[notremoves > 0]
#
#        newnsamp=x_global.shape[0]
#        print('reduced content to ', int(float(newnsamp)/float(self.nsamples)*100),'%')
#
#        print('remove nans')
#        x_global = np.where(np.logical_and(np.isfinite(x_global), (np.abs(x_global) < 100000.0)), x_global, 0)
#        return [x_global], [truth], []
#
#
