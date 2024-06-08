from .test import TestData
from .BaseDataModule import BaseDataModule
from .physio_datamodule import physioDataModule
from .SD_datamodule import SDDataModule
from .Young_datamodule import YoungDataModule
from .SHHS_datamodule import SHHSDataModule
from .SHHS1_datamodule import SHHS1DataModule
from .SHHS2_datamodule import SHHS2DataModule
from .MASS_datamodule import MASSDataModule
from .EDF_datamodule import EDFDataModule
from .ISRUC_datamodule import ISRUCDataModule
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
    'EDF': EDFDataModule,
    'ISRUC_S3': ISRUCDataModule,
    'ISRUC_S1': ISRUCDataModule
}