from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
def add_zoom_inset(ax, zoom,loc, x,y, xlim, ylim , sy = None, 
                   xlabel = '', ylabel = '', label_fs = 18,
                   mark_inset_loc = (3,1), borderpad = 4,
                   bbox_to_anchor = None, bbox_transform = None):
    """Add inset axis that shows a region of the data.
    
    Parameters:
    -----------
    ax: axis object
    zoom: float, zoom factor
    loc: integer, location of inset axis
    x,y, sy: array_like, data to plot
    xlim, ylim: (float, float) limits for x and y axis
    xlabel, ylabel: str, label for x and y axis
    label_fs: float, fonstsize for x and y axis labels
    mark_inset_loc: (int, int), corners for connection to the new axis
    borderpad: float, distance from border
    """
    
    axins = zoomed_inset_axes(ax,zoom,loc = loc,
                              borderpad=borderpad, 
                             bbox_to_anchor=bbox_to_anchor,bbox_transform=bbox_transform) 
    if sy==None:
        axins.plot(x, y)
    else: 
        axins.errorbar(x, y, yerr=sy, fmt='.b',  ecolor='b', elinewidth=.5, 
             capsize=0, capthick=0.1)
        
    if not(ylim==None):
        axins.set_ylim(ylim)
    
    axins.set_xlabel(xlabel, fontsize = label_fs)
    axins.set_ylabel(ylabel, fontsize = label_fs)
    axins.set_xlim(xlim) # Limit the region for zoom
    mark_inset(ax, axins, loc1=mark_inset_loc[0], 
               loc2=mark_inset_loc[1], fc="none", ec="0.5")
    return axins