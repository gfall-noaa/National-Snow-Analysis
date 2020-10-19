import matplotlib
matplotlib.use('AGG')
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import os, shutil
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors

def make_map(lats, lons, data, map_colors, breaks,  breaks_labels, cm_label, title, hucs, save):
    """Create a map image of SNODAS Data and SNODAS Derived Data
    Usage: make_map(lats, lons, data, map_colors, breaks, breaks_labels,
    cm_label, title, hucs, save)
    lats = ncdf4 lats variables object
    lons = ncdf4 lons variables object
    data = ncdf4 data variables object
    map_colors = list of RGB values (range(0,1)) for raster symbology
    breaks = list of values by which to assign colors
    breaks_labels = list of values to use in legend
    cm_label = Legend label
    title = map title
    hucs = yes/no to plot 8 digit HUCS on the map
    save = Filename to save the graphic
    """
    
    # Set up plot parameters
    #plt.style.use('ggplot')
    title_font = {'family':'sans-serif', 'size':16}
    legend_font = {'family':'sans-serif', 'size':12}                 
    fig = plt.figure()
    fig.set_size_inches(17, 11)
    ax=plt.Axes(fig, [0.,0.,1.,1.],)
    fig.add_axes(ax)
    ax.set_title(title, **title_font)
    
    # Create colorbar objects
    cm = ListedColormap(map_colors)
    norm = colors.BoundaryNorm(boundaries=breaks, ncolors=len(map_colors))
    
    # Create a 2D array of lat/long values
    lon_0 = lons.mean()
    lat_0 = lats.mean() + 1.25
    lon, lat = np.meshgrid(lons,lats)
  
    print('Setting Up Basemap')
    # Set up the base map (width/height = meters)
    m = Basemap(width=6000000, height=3500000, 
                resolution='l', projection='stere', 
                lat_ts=40, lat_0=lat_0, lon_0=lon_0)

    xi,yi = m(lon,lat)
    
    # Create the SNODAS Layer for plotting
    m.drawcoastlines(linewidth=0.5)
    m.fillcontinents(color = '0.6', lake_color='#afc6ce', zorder=1)
    m.drawmapboundary(fill_color='#afc6ce')
    cs = m.pcolormesh(xi,yi, np.squeeze(data), norm=norm, cmap=cm, rasterized=True, zorder=2)
    
    # Draw land and water for background
    #m.drawlsmask(land_color='0.4', ocean_color='#afc6ce', lakes=True)
   
    # Draw Lat and Longitude Lines
    m.drawparallels(np.arange(-80., 81., 10.), labels=[0,0,0,0], linewidth=0.25)
    m.drawmeridians(np.arange(-180., 180., 10.), labels=[0,0,0,0], linewidth=0.25)
    
    # Map Background Layers
    m.drawcoastlines(linewidth=0.5)
    m.drawstates(linewidth=0.25)
    m.drawcountries()
    if hucs == 'yes':
        shp_file = ('/net/home/scarter/GIS_Files/HUCs_8')
        m.readshapefile(shp_file, 'HUCs_8', linewidth=0.15)
    
    
    # Create Colorbar Legend
    cbar = plt.colorbar(cs, aspect=15, shrink=0.35, extend='max', pad=-0.12)                
    cbar.set_ticks(breaks)
    cbar.ax.set_yticklabels(breaks_labels)
    cbar.set_label(cm_label, **legend_font)
    cbar.ax.tick_params(labelsize=11)
    
    print('Saving Figure')
    # Save the plot
    plt.savefig(save, bbox_inches='tight')
    
    # Using imagemagick, put the NOAA logo on the map
    expr = 'convert %s /net/home/scarter/SNODAS_Development/SWEpy/noaa1.png -flatten %s' % (save, save)
    os.system(expr)
    
    # Delete Plot
    plt.close(fig)
