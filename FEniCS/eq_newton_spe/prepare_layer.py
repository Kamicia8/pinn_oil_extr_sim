import numpy as np

NX, NY, NZ = 60, 220, 85
PATH_INPUT = r'/mnt/c/PINN_mgr/spe_perm_model2.dat'
PATH_OUTPUT = 'spe_model2_layer50.npy'

def extract_and_save_layer(layer_idx=50):
    raw_data = np.fromfile(PATH_INPUT, sep=' ')
    
    kx_all = raw_data[0 : NX*NY*NZ]
    
    kx_3d = kx_all.reshape((NZ, NY, NX))
    
    layer = kx_3d[layer_idx, :, :]
    
    np.save(PATH_OUTPUT, layer)
    print(f"Warstwa {layer_idx} zapisana w {PATH_OUTPUT}")
    print(f"Kształt macierzy: {layer.shape}")

if __name__ == "__main__":
    extract_and_save_layer()