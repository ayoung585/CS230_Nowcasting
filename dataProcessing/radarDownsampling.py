from joblib import Parallel, delayed
from skimage.transform import downscale_local_mean
import h5py

def downsample_file(file_path):
    with h5py.File(file_path, 'r+') as f:
        data = f['your_dataset_name'][:]
        data_downsampled = downscale_local_mean(data, (1, 2, 2))
        del f['your_dataset_name']
        f.create_dataset('your_dataset_name', data=data_downsampled)


path = '/home/ubuntu/data/TAASRAD19/hdf_archives/'
file_paths = [path+'20120802.hdf5', path+'20120803.hdf5', path+'20120804.hdf5'] # list of HDF5 file paths
Parallel(n_jobs=-1)(delayed(downsample_file)(file_path) for file_path in file_paths)