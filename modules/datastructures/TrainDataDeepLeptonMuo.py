from DeepJetCore.TrainData import TrainData
from DeepJetCore.TrainData import fileTimeOut as tdfto
import numpy as np

#set tree name to use
import DeepJetCore.preprocessing
DeepJetCore.preprocessing.setTreeName('tree')

def fileTimeOut(fileName, timeOut):
    tdfto(fileName, timeOut)

from TrainDataDeepLepton import TrainDataDeepLepton

class TrainDataDeepLeptonMuo(TrainDataDeepLepton):
    '''
    Muon Class for DeepLepton Training Data.
    '''

    def __init__(self):
        super().__init__()
    
        self.global_branches = [
            #'lep_pt', 'lep_eta',
            'lep_phi',
            'lep_mediumId',
            'lep_miniPFRelIso_all',
            'lep_sip3d', 'lep_dxy', 'lep_dz',
            'lep_charge',
            'lep_dxyErr', 'lep_dzErr', 'lep_ip3d',
            'lep_jetPtRelv2', 'lep_jetRelIso',
            'lep_miniPFRelIso_chg', 'lep_mvaLowPt', 'lep_nStations', 'lep_nTrackerLayers', 'lep_pfRelIso03_all', 'lep_pfRelIso03_chg', 'lep_pfRelIso04_all', 'lep_ptErr',
            'lep_segmentComp', 'lep_tkRelIso', 'lep_tunepRelPt',
            ]
    
