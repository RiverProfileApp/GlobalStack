import numpy as np
from globalstack_functions import sinkfill
import rasterio as rio
Z = np.squeeze(np.float64(rio.open('/Volumes/T7/eu/dem2.tif').read()))
Z = sinkfill(Z)
np.savez_compressed('euz2',Z)
