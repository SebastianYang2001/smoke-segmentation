import os
from tqdm import tqdm
import s3fs
import numpy as np
import rioxarray
from PIL import Image

def get_double_goes_data(download_dir, satellite, year=2021, doy=201, hour=0, tag="C"):
    bucket_name = 'noaa-'+str(satellite).lower()
    product_name = 'ABI-L2-MCMIP'+tag
    
    fs = s3fs.S3FileSystem(anon=True)

    # Write prefix for the files of interest, and list all files beginning with this prefix.
    prefix = f'{bucket_name}/{product_name}/{year}/{doy:03.0f}/{hour:02.0f}/'
    try: 
        files = fs.ls(prefix)
    except:
        return None
    if files:
        files_to_download = [files[0], files[len(files)//2]]
        paths = []
        for file in files_to_download:
            #Read the netCDF files in the list and download
            path = os.path.join(download_dir, file.split("/")[-1])
            fs.download(file, path)
            paths.append(path)
        return paths
    else:
        return None
def open_netcdf(filepath):
    C = rioxarray.open_rasterio(filepath)
    return C

#contstruct true color
def get_RGB(C, gamma=2.2, night=True):
    #load the correct channels
    R = C['CMI_C02'].data 
    G = C['CMI_C03'].data 
    B = C['CMI_C01'].data 
    if len (R.shape)>2:
        R = R[0] * C["CMI_C02"].scale_factor
        G = G[0] * C["CMI_C03"].scale_factor
        B = B[0] * C["CMI_C01"].scale_factor
    #apply range limits for each channel. RGB values must be between 0 and 1
    R = np.clip(R, 0, 1)
    G = np.clip(G, 0, 1)
    B = np.clip(B, 0, 1)
    #apply a gamma correction to the image to correct ABI detector brightness
    R = np.power(R, 1/gamma)
    G = np.power(G, 1/gamma)
    B = np.power(B, 1/gamma)
    # Calculate the "True" Green
    G_true = 0.45 * R + 0.1 * G + 0.45 * B
    G_true = np.clip(G_true, 0, 1)  # apply limits again, just in case.
    # The RGB array for the true color image
    RGB = np.dstack([R, G_true, B])
    
    if night == True:
        cleanIR = C['CMI_C13'].data[0] * C['CMI_C13'].scale_factor
        # Normalize the channel between a range.
        #       cleanIR = (cleanIR-minimumValue)/(maximumValue-minimumValue)
        cleanIR = (cleanIR-90)/(313-90)
        # Apply range limits to make sure values are between 0 and 1
        cleanIR = np.clip(cleanIR, 0, 1)
        # Invert colors so that cold clouds are white
        cleanIR = 1 - cleanIR
        # Lessen the brightness of the coldest clouds so they don't appear so bright
        # when we overlay it on the true color image.
        cleanIR = cleanIR/1.4
        RGB_ColorIR = np.dstack([np.maximum(R, cleanIR), np.maximum(G_true, cleanIR),
                            np.maximum(B, cleanIR)])
        C.close()
        return RGB, RGB_ColorIR
    
    else:
        return RGB

def contrast_correction(RGB, contrast):
    """
    Modify the contrast of an RGB
    See:
    https://www.dfstudios.co.uk/articles/programming/image-programming-algorithms/image-processing-algorithms-part-5-contrast-adjustment/

    Input:
        color    - an array representing the R, G, and/or B channel
        contrast - contrast correction level
    """
    F = (259*(contrast + 255))/(255.*259-contrast)
    COLOR = F*(RGB-.5)+.5
    COLOR = np.clip(COLOR, 0, 1)  # Force value limits 0 through 1.
    return COLOR

def save(RGB, filepath, ftype='jpg'):
    maxval = np.max(RGB)
    minval = np.min(RGB)
    img = 255 * (RGB - minval)/(maxval - minval)
    img = Image.fromarray(img.astype(np.uint8))
    img.save(filepath.replace(".nc", ".png"))
    return

def build_pretraining_dataset(output_dir, years, start_doy, end_doy):
    hours_16 = [i for i in range(16, 22)]
    hours_17 = [i for i in range(19, 24)] + [0]
    hours = {"goes17": hours_17, "goes16": hours_16, "goes18": hours_17}
    netcdf_dir = os.path.join(output_dir, "nc")
    png_dir = os.path.join(output_dir, "png")
    satellites = ["goes17", "goes16"]
    #os.makedirs(netcdf_dir)
    #os.makedirs(png_dir)
    for year in tqdm(years):
        if year == 2023:
            satellites = ["goes18", "goes16"]
        if year == 2022:
            start = 228
        else:
            start = start_doy
        for d in tqdm(range(start, end_doy)):
            for satellite in satellites:
                for hour in hours[satellite]:
                    files = get_double_goes_data(netcdf_dir, satellite=satellite, year=year, doy=d, hour=hour, tag="C")
                    if files:
                        for file in files: 
                            file = os.path.basename(file)
                            C = open_netcdf(os.path.join(netcdf_dir,file))
                            RGB = get_RGB(C, night=False)
                            RGB_contrast = contrast_correction(RGB, 105)
                            save(RGB_contrast, os.path.join(png_dir,file))
                            C.close()
                            del C
                            os.remove(os.path.join(netcdf_dir,file))

    return

if __name__ == "__main__":
    output_dir = os.path.join("..", "..", "..", "media", "FS2", "data", "pretraining_data")
    years = [2023]#[2020, 2021, 2022, 2023]
    build_pretraining_dataset(output_dir, years, start_doy=167, end_doy=258)