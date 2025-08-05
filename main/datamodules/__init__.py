from .test import TestData
from .BaseDataModule import BaseDataModule
from .physio_datamodule import physioDataModule
from .SD_datamodule import SDDataModule
from .Young_datamodule import YoungDataModule
from .SHHS_datamodule import SHHSDataModule
from .SHHS1_datamodule import SHHS1DataModule
from .SHHS2_datamodule import SHHS2DataModule
from .MASS_datamodule import MASSDataModule
from .EDF_datamodule import EDFDataModule, EDFAugDataModule
from .ISRUC_datamodule import ISRUCDataModule
from .MGH_datamodule import MGHDataModule
from .Apnea_datamodule import ApneaDataModule
from .CAP_datamodule import CAPDataModule
from .UMS_datamodule import UMSDataModule
_datamodules = {
    'physio_train': physioDataModule,
    'physio_test': physioDataModule,
    'SD': SDDataModule,
    'Young': YoungDataModule,
    'SHHS': SHHSDataModule,
    'SHHS1': SHHS1DataModule,
    'SHHS2': SHHS2DataModule,
    'MASS': MASSDataModule,
    'MASS1': MASSDataModule,
    'MASS2': MASSDataModule,
    'MASS3': MASSDataModule,
    'MASS4': MASSDataModule,
    'MASS5': MASSDataModule,
    'MASS2_AUG': MASSDataModule,
    'EDF': EDFDataModule,
    'EDF_AUG': EDFAugDataModule,
    'ISRUC_S3': ISRUCDataModule,
    'ISRUC_S1': ISRUCDataModule,
    'MGH': MGHDataModule,
    'MASS_Apnea': ApneaDataModule,
    'CAP_n': CAPDataModule,
    'CAP_ins': CAPDataModule,
    'CAP_narco': CAPDataModule,
    'CAP_nfle': CAPDataModule,
    'CAP_plm': CAPDataModule,
    'CAP_rbd': CAPDataModule,
    'CAP_sdb': CAPDataModule,
    'UMS': UMSDataModule
}