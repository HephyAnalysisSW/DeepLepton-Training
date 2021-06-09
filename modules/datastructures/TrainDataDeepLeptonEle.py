from DeepJetCore.TrainData import TrainData
from DeepJetCore.TrainData import fileTimeOut as tdfto
import numpy as np


from TrainDataDeepLepton import TrainDataDeepLepton

class TrainDataDeepLeptonEle(TrainDataDeepLepton):
    '''
    Electron Class for DeepLepton Training Data.
    '''

    def __init__(self):
        super().__init__()
        self.global_branches = [
            #'lep_pt',
            #'lep_eta',
            'lep_phi',
            'lep_pdgId',
            'lep_cutBased',
            'lep_miniPFRelIso_all',
            'lep_pfRelIso03_all',
            'lep_sip3d',
            'lep_lostHits',
            'lep_convVeto',
            'lep_dxy',
            'lep_dz',
            'lep_charge',
            'lep_deltaEtaSC',
            'lep_vidNestedWPBitmap',
            'lep_dr03EcalRecHitSumEt',
            'lep_dr03HcalDepth1TowerSumEt',
            'lep_dr03TkSumPt',
            'lep_dxyErr',
            'lep_dzErr',
            'lep_eCorr',
            'lep_eInvMinusPInv',
            'lep_energyErr',
            'lep_hoe',
            'lep_ip3d',
            'lep_jetPtRelv2',
            'lep_jetRelIso',
            'lep_miniPFRelIso_chg',
            'lep_mvaFall17V2noIso',
            'lep_pfRelIso03_chg',
            'lep_r9',
            'lep_sieie',
            #'lep_genPartFlav',
            #'npfCand_charged',
            #'npfCand_neutral',
            #'npfCand_photon',
            #'npfCand_electron',
            #'npfCand_muon',
            #'nSV',       
            ]
    
