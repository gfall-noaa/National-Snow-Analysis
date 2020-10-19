"""
United States Department of Commerce
National Oceanic and Atmospheric Administration
National Weather Service
Office of Water Prediction


@author: Shawn M. Carter, UCAR/CPAESS

Script requires ability to access OWP Chanhassen
development workstations to retrieve archived
SNODAS model output files.

This script creates SNODAS Period of Record Norms Graphs
that are publicly accessed at http://www.nohrsc.noaa.gov/normals

Input:
    date - (str) Y-m-d

Version 3: 2018-05-30
"""
# Base python imports
from datetime import datetime
import glob
import gzip
import os
from shutil import copyfileobj, copyfile
import sys
import tarfile
#from urllib.request import urlopen
from urllib2 import urlopen, URLError

# Extra library imports
from netCDF4 import Dataset
import gdal
import numpy as np
from scipy import ndimage

# Local imports
sys.path.append( '/net/home/scarter/SNODAS_Development/SWEpy' )
import nohrsc_colors as nc
import SNODAS_map_maker as mm

# Global Variables
tmp_dir = '/net/tmp'
os.chdir(tmp_dir)

def _get_if_exists( data, key ):
    '''Yields value from dictionary based on  key, avoids key errors.'''
    if key in data:
        return data[ key ]

def get_rasters_web( model_date ):
    #model_date = model_date.strptime('%Y%m%d')
    print(model_date)
    date = model_date[ :-2 ]
    year = model_date[ 0:4 ]
    mn = model_date[ 4:6 ]
    month = {'01': '01_Jan', '02': '02_Feb', '03': '03_Mar', '04': '04_Apr',
             '05': '05_May', '06': '06_Jun', '07': '07_Jul', '08': '08_Aug',
             '09': '09_Sep', '10': '10_Oct', '11': '11_Nov', '12': '12_Dec'}

    url = ( 'ftp://sidads.colorado.edu/DATASETS/NOAA/G02158/masked/'
            '{}/{}/SNODAS_{}.tar'.format( year, month[mn], model_date ))
    file_name = url.split('/')[-1:][0]
    file_name_path = os.path.join(tmp_dir, file_name)
    try:
        response = urlopen(url)
    except URLError:
        return 1, 2, 3, 4, 5
    with open(file_name_path, 'wb') as out_file:
        copyfileobj(response, out_file)

    tar = tarfile.open(file_name_path)
    tar.extractall(path = tmp_dir)
    tar.close()
    os.remove(file_name_path)

    for f in glob.glob( 'us_ssmv11034*{}*'.format(model_date)):
        in_f = gzip.open(f, 'rb')
        in_f_contents = in_f.read()
        with open( f[:-3], 'wb') as out_f:
           out_f.write( in_f_contents )
        in_f.close()
        os.remove(f)

    data_units = {}
    
    for f in glob.glob('us_ssmv*{}*.dat'.format(model_date)):
        old_dat = f[:-3] + 'dat'
        new_dat = f[:-3] + 'bil'
        os.rename(old_dat, new_dat)

    for f in glob.glob('us_ssmv*{}*.txt'.format(model_date)):
        if 'ssmv11034' in f:
            hdr_values = {}
            new_hdr = f[:-3] + 'hdr'
            root_name = f[:-4]
            with open(f, 'r') as old_header:
                for row in old_header:
                    
                    hdr_values[row.split(':')[0]] = row.split(':')[1].strip()
                data_units[root_name] = hdr_values['Data units']
                if data_units[root_name] == 'Kelvins':
                    data_units[root_name] = 'Kelvins / 1'

            hdr_contents = { '1': 'byteorder M', '2': 'layout bil',
                         '3': 'nbands 1', '4': 'nbits 16',
                         '5': 'ncols {}'.format(hdr_values['Number of columns']),
                         '6': 'nrows {}'.format(hdr_values['Number of rows']),
                         '7': 'ulxmap {}'.format(hdr_values['Benchmark x-axis coordinate']),
                         '8': 'ulymap {}'.format(hdr_values['Benchmark y-axis coordinate']),
                         '9': 'xdim {}'.format(hdr_values['X-axis resolution']),
                         '10': 'ydim {}'.format(hdr_values['Y-axis resolution'])}
        with open(new_hdr, 'w') as out_file:
            for i in range(1, len(hdr_contents) + 1):
                out_file.write(hdr_contents[str(i)] + '\n')
        os.remove(f)

        for f in glob.glob('us_ssmv11034*{}*.bil'.format(model_date)):

            ds = gdal.Open(f)
            band = ds.GetRasterBand(1)
            array = band.ReadAsArray()
            array = array * 0.001
            gt = ds.GetGeoTransform()   
            nrows = array.shape[0]
            ncols = array.shape[1]
            scalar = data_units['us_ssmv11034tS__T0001TTNATS{}05HP001'.format(model_date)]
            ndv = 55537 * 0.001
            array[array == ndv] = -9999.
    for f in glob.glob('us_ssmv*'):
        os.remove(f)
    return array, nrows, ncols, scalar, ndv
    
def get_rasters( model_date ):
    '''Yields a numpy array of SNODAS SWE and metadata derived
    from header file.
    '''
    
    file_dir = '/net/lfs0data5/NSIDC_archive/masked/1034'+'/{}/{}'.format(model_date[0:4], model_date[4:6])
    collaborators_dir = '/net/ftp/products/collaborators'
    file_template = 'us_ssmv11034tS__T0001TTNATS{date}05HP001.{{ext}}'.format( date=model_date )
    dat_file = file_template.format( ext='dat' )
    hdr_file = file_template.format( ext='Hdr' )
    gz_file = file_template.format( ext='Hdr.gz' )

    # Find model results in collborators or NSIDC Archive
    gz_file_path = os.path.join( file_dir, gz_file )
    grz_file_path = os.path.join( collaborators_dir, dat_file[ :-4 ] + '.grz' )
    print(gz_file_path)
    if os.path.isfile( gz_file_path ):
        print(gz_file_path)
        files = [ dat_file, hdr_file ]
        for zipped_file in files:
            zipped_file_path = os.path.join( file_dir, zipped_file + '.gz' )
            if os.path.isfile( zipped_file_path ):
                with gzip.open( zipped_file_path, 'rb' ) as in_file:
                    in_file_contents = in_file.read()
                    tmp_zip = os.path.join( tmp_dir, zipped_file )
                    with open( tmp_zip, 'wb' ) as out_file:
                        out_file.write( in_file_contents )

    elif os.path.isfile( grz_file_path ):
        print(grz_file_path)
        with tarfile.open( grz_file_path ) as tar_file:
            tar_file.extractall( path='/net/tmp' )
    else:
        array, nrows, ncols, scalar, ndv = get_rasters_web(model_date)
        return array, nrows, ncols, scalar, ndv
        
    # Dictionary of data units to scale the output tiffs
    header = {}
    hdr_file_path = os.path.join( tmp_dir, hdr_file )
    with open( hdr_file_path, 'r' ) as hdr_file:
        for row in hdr_file:
            header[ row.split( ':' )[ 0 ]] = row.split( ':' )[ 1 ].strip()

    nrows = int( _get_if_exists( header, 'Number of rows' ))
    ncols = int( _get_if_exists( header, 'Number of columns' ))
    scalar = float( _get_if_exists( header, 'Data units' ).split( '/' )[ 1 ].strip())
    if not _get_if_exists( header, 'Data intercept' ):
        intercept = 0.0
    else:
        intercept = float( _get_if_exists( header, 'Data intercept' ))
    ndv = float( _get_if_exists( header, 'No data value' ))

    # Read and scale the array (dat file values stored as little endian 4-bit binary)
    dat_file_path = os.path.join( tmp_dir, dat_file )
    array = np.fromfile( dat_file_path, dtype=np.dtype( '>i2' ))
    array_float = array.astype( 'float' )
    array_scaled = np.where( array_float != ndv, ( array_float/scalar ) + intercept, -9999. )
    array_shaped = array_scaled.reshape( nrows, ncols )

    # Delete temporary files from /net/tmp
    for dl_file in glob.glob( os.path.join( tmp_dir, 'us_ssmv11034*' )):
        os.remove( dl_file )
    return array_shaped, nrows, ncols, scalar, ndv

def get_huc8():
    '''Yields numpy array of huc8s, lats, lons, and huc8 numbers
    from NHD V2 HUC8's file that has previously been converted to a
    raster and stored as a netCDF.
    '''

    huc8_dir = ( '/net/home/scarter/GIS_Files/huc8.nc' )
    with  Dataset( huc8_dir ) as huc8_file:
        huc8 = huc8_file[ 'Band1' ][:]
        lats = huc8_file[ 'lat' ][:]
        lons = huc8_file[ 'lon' ][:]
    huc8 =  huc8.astype( int )
    huc8_index = list( np.unique( huc8 ))
    return np.flipud(huc8), lats, lons, huc8_index

def date_range( start_date ):
    '''Yields a list of years in the SNODAS record based on the date.
    SNODAS output prior to Oct 2005 are un-assimilated and therefore
    not likely accurate.
    '''
    month = start_date.month
    year = start_date.year
    if month > 9:
        years = range( 2004, year, 1 )
    else:
        years = range( 2005, year, 1 )
    return years

def snodas_mask( array, ndv ):
    '''Yields a binary mask wherever SNODAS calculated NDV in model.
    At different times in the model's history, areas have been added
    to avoid processing such as lakes, reservoirs, glaciers, and large
    rivers.  Array is a stack of SNODAS model arrays representing the
    full time domain for this model run.
    '''
    for i, values  in enumerate( array ):
        arr = np.copy( values[:])
        arr_masked = np.where( arr == ndv, 1.0, 0.0 )
        if i == 0:
            mask = arr_masked[:]
        else:
            mask += arr_masked[:]
    mask_aggregate = np.where( mask > 0, 0, 1 )
    return mask_aggregate

def calculate_regular_snow( array ):
    '''Creates a binary mask to perform normals calculations only on
    those arrays that have snow > 50% of record.  This mask avoids creating
    small means of snow in places in that have had unlikely snow amounts
    during the period of record.  For example, Texas-Georgia in early December.
    '''
    snow_probability = 0.5
    for i, values in enumerate( array ):
        arr = np.copy( values[:])
        arr_masked = np.where( arr > 0.0, 1.0, 0.0 )
        if i == 0:
            days_in_record = arr_masked[:]
        else:
            days_in_record += arr_masked[:]
    days_in_record_aggregate = np.where( days_in_record > int( snow_probability * len( array )), 1, 0 )
    return days_in_record_aggregate

def zonal_sum( swe, huc, labels ):
    '''Yields an array representing  the mean of swe for each HUC8.'''
    #huc = np.where(huc==-9999., 0, huc)
    swe = np.where(swe==-9999., 0, swe)
    sums = ndimage.sum( swe, huc, labels )
    sums_dict = dict( zip( labels, sums))
    sums_dict[0] = -9999.
    sums_arr = np.vectorize( sums_dict.get )( huc )

    return sums_arr

def zonal_count(swe, huc, labels):
    #huc = np.where(huc==-9999., 0, huc)
    swe = np.where(swe >= 0, 1, 0)
    sums = ndimage.sum(swe, huc, labels)
    sums_dict = dict(zip(labels, sums))
    sums_dict[0] = -9999.
    sums_arr = np.vectorize(sums_dict.get)(huc)
    return sums_arr

def percent_difference( normals, current ):
    '''Yields an array representing percent difference and controls for division
    by zero errors.

    Arrays need to be multiplied by the output of the snodas_mask
    function prior to being input into this function.
    '''
    threshold = 0.005
    per_diff = np.where(
                       ( current >= threshold ) & ( normals >= threshold ),
                       100 * np.divide( current, normals ),
                       np.where(
                               ( current >= threshold ) & ( normals <= threshold ),
                               99999, -9999
                                )
                        )

    return per_diff

def percent_difference_hucs(normals, current, sums):
    current = np.where(current/sums > 0.001, current, -9999.)
    normals = np.where(normals/sums > 0.001, normals, -9999.)

    per_diff = np.where(
        (current >= 0.001) & (normals >= 0.001),
        100 * np.divide(current, normals),
        np.where(
            (current >= 0.001) & (normals < 0.001),
            99999, -9999))
    #per_diff = np.where(normals == 0, -9999, per_diff)
    return per_diff

def calculate_difference( normals, current ):
    '''Yields an array describing the difference between period of record and current snow'''
    metric_to_us_scale = 25.4
    diff = np.where(
                   ( normals == -9999. ) | ( current == -9999. ), -9999.,
                   np.where(
                           (current == 0) & (normals == 0), -499999.,
                           (current - normals) / metric_to_us_scale
                           )
                   )
    return diff

def main( start_date ):
    ''' Main function, start_date is datetime object '''
    
    # Acquire data and format date strings
    huc, lats, lons, labels = get_huc8()
    years = date_range( start_date )
    month_day = start_date.strftime( '%b %d' )
    graphic_name_template = 'SNODAS_{{type}}_{date}.png'.format(date = start_date.strftime('%Y%m%d'))
    year_range = '{} - {} {}'.format( years[0], years[-1], month_day )
    save_folder = start_date.strftime('%Y%m')
    sav_dir = os.path.join('/net/tmp', save_folder)
    for i in range( len( years )):
        model_date = '{}{}{}'.format(years[i], '%02d' % start_date.month, start_date.strftime('%d') )
        year_array, nrows, ncols, scalar, ndv = get_rasters( model_date )
        if isinstance(year_array, (np.ndarray)):
            
            if i == 0:
                norms = [ year_array ]
            else:
                norms.append( year_array )
    
    regular_snow = calculate_regular_snow( norms )
    mask = snodas_mask( norms, ndv )
    start_date = start_date.strftime('%Y%m%d')
    current_swe, nrows, ncols, scalar, ndv = get_rasters( start_date )

    # Create  HUC Percent Difference graphic
    print('Creating HUC Percent Difference Graphic.')
    swe_mean =  np.median(norms, axis=0)
    swe = np.where(mask > 0, current_swe, -9999.)
    swe = np.where(regular_snow == 0 , -9999., swe)
    swe = np.where(swe == ndv, -9999., swe)
    print(np.nanmax(swe))
    current_huc_swe = zonal_sum( swe, huc, labels )
    mean_huc_swe = zonal_sum(swe_mean, huc, labels)
    count_hucs = zonal_count(swe, huc, labels)
    percent_diff_hucs = percent_difference_hucs( mean_huc_swe, current_huc_swe, count_hucs)
    percent_diff_hucs = np.ma.masked_where( huc == 0, percent_diff_hucs )
    percent_diff_hucs = percent_diff_hucs * mask
    percent_diff_hucs =  np.flipud(percent_diff_hucs )
    colors, breaks, labels = nc.get_colors( 'per_diff', 'percent' )
    title = 'SNODAS SWE, Percent of ' + str( len( years )) + ' Year Median, ' + year_range
    save_file_name = graphic_name_template.format( type = 'SWE_percent_normal_HUC8' )
    save_file = os.path.join(tmp_dir, save_file_name)
    mm.make_map( lats, lons, percent_diff_hucs, colors, breaks, labels,
                 'Percent of Median', title, 'yes', save_file )

    # Create HUC Difference graphic
    #print 'Creating HUC Difference Graphic.'
    #diff_hucs = calculate_difference( huc_mean, current_huc_swe )
    #diff_hucs = np.ma.masked_where( huc==0, diff_hucs )
    #diff_hucs = np.flipud( diff_hucs )
    #colors, breaks, labels = nc.get_colors( 'swe_delta', 'delta_meters' )
    #title = 'SNODAS SWE, Difference from ' + str( len( years )) + ' Year Mean, ' + year_range
    #save_file_name = graphic_name_template.format( type = 'SWE_diff_from_normal_HUC8' )
    #save_file = os.path.join(tmp_dir, save_file_name)
    #mm.make_map(lats, lons, diff_hucs, colors, breaks, labels,
    #           'Difference in Inches of Water',
    #           title, 'yes',
    #           save_file )

    
    # Create Daily Graphic
    print('Creating the Daily SWE Graphic')
    colors, breaks, labels = nc.get_colors( 'swe', 'meters' )
    title = 'SNODAS SWE, Snow Water Equivalent, ' + date.strftime('%Y%m%d')
    save_file_name = graphic_name_template.format( type = 'SWE' )
    save_file = os.path.join(tmp_dir, save_file_name)
    swe = np.ma.masked_where( huc==0, current_swe )
    swe_flipped =  np.flipud(swe)
    mm.make_map( lats, lons, swe_flipped, colors, breaks, labels,
                 'Snow Water Equivalent',
                 title, 'no',
                 save_file )
     
    # Create Normals Graphic
    print('Creating the SWE Normals Graphic')
    #for i, values in enumerate( norms ):
    #    arr = values * regular_snow * mask
    #    if i == 0:
    #        norms_mean_arrays = arr
    #    else:
    #        norms_mean_arrays += arr
    norms_mean = np.median(norms, axis=0, overwrite_input=True)
    #norms_mean = norms_mean_arrays / len( norms )
    norms_mean_masked = np.ma.masked_where( huc==0, norms_mean )
    norms_mean_masked = norms_mean_masked * mask
    norms_mean_flipped = np.flipud(norms_mean_masked )
    title = 'SNODAS SWE, ' + str( len( years )) + ' Year Median, Snow Water Equivalent, ' + year_range
    save_file_name = graphic_name_template.format( type = 'SWE_normal' )
    save_file = os.path.join( tmp_dir, save_file_name)
    mm.make_map( lats, lons, norms_mean_flipped, colors, breaks, labels,
                'Snow Water Equivalent',
                title, 'no',
                save_file )
    
    # Create the Delta SWE Graphic
    print('Creating the Delta SWE Graphic.')
    colors, breaks, labels = nc.get_colors( 'swe_delta', 'delta_meters' )
    #for i, values in enumerate( norms ):
    #    arr = values * mask * regular_snow
    #    if i == 0:
    #        norm_swe = arr
    #    else:
    #        norm_swe += arr
    #norm_swe_mean = norm_swe / len( norms )
    #norms_swe_mean = np.median(norms, axis=0)
    norms_mean = np.where(swe == -9999,  -9999., norms_mean)
    diff_mean = calculate_difference( norms_mean, swe )
    diff_mean_masked = np.ma.masked_where( huc==0, diff_mean )
    diff_mean_masked = np.ma.masked_where(diff_mean_masked == -9999.0, diff_mean_masked)
    diff_mean_flipped = np.flipud(diff_mean_masked )
    title = 'SNODAS SWE, Difference from ' + str( len( years )) + ' Year Median, ' + year_range
    save_file_name = graphic_name_template.format( type = 'SWE_diff_from_normal' )
    save_file = os.path.join(tmp_dir, save_file_name )
    mm.make_map( lats, lons, diff_mean_flipped, colors, breaks, labels,
                'Difference in Inches of Water',
                title,
                'no', save_file )

    # Create the percent difference graphic
    print('Creating the Percent Difference Graphic.')
    #for i, values in enumerate( norms ):
    #    arr = values * regular_snow * mask
    #    if i == 0:
    #        norms_mean_arrays = arr
    #    else:
    #        norms_mean_arrays += arr
    #norms_mean = norms_mean_arrays / len( norms )
    #norms_mean = np.median(norms, axis=2)
    #swe = current_swe * mask
    per_diff = percent_difference( norms_mean, swe)
    per_diff_masked = np.ma.masked_where( huc==0, per_diff )
    per_diff_flipped =  np.flipud(per_diff_masked )
    colors, breaks, labels = nc.get_colors( 'per_diff', 'percent' )
    title = 'SNODAS SWE, Percent of ' + str( len( years )) + ' Year Median, ' + year_range
    save_file_name = graphic_name_template.format( type = 'SWE_percent_normal' )
    save_file = os.path.join( tmp_dir, save_file_name )
    mm.make_map( lats, lons, per_diff_flipped, colors, breaks, labels,
                'Percent of Median',
                title,
                'no', save_file )

    
if __name__ == '__main__':
    if len( sys.argv ) > 1:
        date = datetime.strptime( sys.argv[ 1 ], '%Y%m%d' )
    else:
        date = datetime.now()
    main( date )
