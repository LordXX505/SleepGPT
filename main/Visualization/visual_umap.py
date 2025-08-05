from utils import UMAPPlotter
import os

plotter = UMAPPlotter()  # 设置UMAP迭代次数和初始化

label_to_color = {
        0: '#89c8e8',
        1: '#ebd57c',
        2: '#1f78b4',
        3: '#6a3d9a',
        4: '#8acc72',
    }
def shhs():
    print('start: shhs')
    checkpoint_dir = f'../../result/SHHS1/UMAP/0/portion_1/last.ckpt'
    save_dir = f'../../result/SHHS1/UMAP_R/'
    os.makedirs(save_dir, exist_ok=True)

    plotter.transform_and_plot(checkpoint_dir, save_dir, label_to_color, down=5, plot_predicted=True, s=0.005)
def EDF_2013():
    checkpoint_dir = f'../../result/EDF/UMAP/1/portion_1/ModelCheckpoint-epoch=40-val_acc=0.8360-val_score=5.3603.ckpt'
    save_dir = f'../../result/EDF/UMAP_R/'
    os.makedirs(save_dir, exist_ok=True)
    plotter = UMAPPlotter()  # 设置UMAP迭代次数和初始化

    plotter.transform_and_plot(checkpoint_dir, save_dir, label_to_color, plot_predicted=True)

def EDF_TCC():
    checkpoint_dir = f'../../result/EDF/UMAP/1/edf_2013_TCC_umap/ModelCheckpoint-epoch=40-val_acc=0.8960-val_macro=0.8253-val_score=6.7213.ckpt'
    save_dir = f'../../result/EDF/UMAP_TCC_R/'
    os.makedirs(save_dir, exist_ok=True)
    plotter = UMAPPlotter()  # 设置UMAP迭代次数和初始化

    plotter.transform_and_plot(checkpoint_dir, save_dir, label_to_color, plot_predicted=True)
def PHY():
    checkpoint_dir = f'../../result/physio_train/UMAP/2/PHY_UMAP/ModelCheckpoint-epoch=99-val_acc=0.8200-val_score=5.3237.ckpt'
    save_dir = f'../../result/physio_train/UMAP/'
    os.makedirs(save_dir, exist_ok=True, )
    plotter = UMAPPlotter()  # 设置UMAP迭代次数和初始化

    plotter.transform_and_plot(checkpoint_dir, save_dir, label_to_color, plot_predicted=True, s=0.05)
# shhs()

def EDF_2018():
    checkpoint_dir = f'../../result/EDF/UMAP/1/edf_2018_1_umap/ModelCheckpoint-epoch=27-val_acc=0.8650-val_score=5.4750.ckpt'
    save_dir = f'../../result/EDF/2018_UMAP_R/'
    os.makedirs(save_dir, exist_ok=True)
    plotter = UMAPPlotter()  # 设置UMAP迭代次数和初始化

    plotter.transform_and_plot(checkpoint_dir, save_dir, label_to_color, plot_predicted=True)

def MASS():
    checkpoint_dir = f'../../result/MASS5/UMAP/14/MASS_UMAP'
    save_dir = f'../../result/MASS/UMAP/'
    os.makedirs(save_dir, exist_ok=True)
    plotter = UMAPPlotter()  # 设置UMAP迭代次数和初始化

    plotter.transform_and_plot(checkpoint_dir, save_dir, label_to_color, plot_predicted=True)
if __name__ == '__main__':
    EDF_2013()
    PHY()
    EDF_TCC()
    shhs()
    EDF_2018()
    MASS()