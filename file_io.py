import h5py
def data_from_h5file(file):
        with h5py.File(file, 'r') as hf:
            return hf[list(hf.keys())[0]][:]