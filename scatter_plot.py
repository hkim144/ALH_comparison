#----------------------------------------------------------------------------
def scatter_plot(ax, in_x_data, in_y_data, **kwargs):
    ''' 
    Function to create scatter plots
    '''
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    import scipy
    from scipy import stats
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                   AutoMinorLocator)

    stdOn	     = kwargs.get('stdOn', False)
    regress_line = kwargs.get('regress_line', True)
    one2one_line  = kwargs.get('one2one_line', True)
    ee_line1  = kwargs.get('ee_line1', False)
    ee_line2  = kwargs.get('ee_line2', False)

    label_p		= kwargs.get('label_p', None)	
    yerr		= kwargs.get('yerr', 0)
    xerr		= kwargs.get('xerr', 0)
    markersize	= kwargs.get('markersize', 0.5)
    elinewidth	= kwargs.get('elinewidth', 1)
    capsize		= kwargs.get('capsize', 2)
    color		= kwargs.get('fcolor', 'black')
    statOn		= kwargs.get('statOn', True)
    fmt			= kwargs.get('fmt', 'o')
    fsize    = kwargs.get('fsize', 12)
    ylabel	    = kwargs.get('ylabel', '')
    xlabel	    = kwargs.get('xlabel', '')
    rmse_coef_flag	 = kwargs.get('rmse_coef_flag', True)
    reg_color	 = kwargs.get('reg_color', 'r')
    eeOn  	    = kwargs.get('eeOn', False)
    eeBias 	    = kwargs.get('eeBias', 0.1)
    eeFrac 	    = kwargs.get('eeFrac', 0.15)
    alpha 	    = kwargs.get('alpha', 1)
    case_str 	= kwargs.get('case_str', None)

    dp_x        = kwargs.get('dp_x', 'x')
    dp_y        = kwargs.get('dp_x', 'y')


    xRange	= kwargs.get('xRange', None)
    yRange	= kwargs.get('yRange', None)


    delta_ticks_X = kwargs.get('delta_ticks_X', None)
    delta_ticks_Y = kwargs.get('delta_ticks_Y', None)

    model = kwargs.get('model', 'OLS')

    # copy and flatten data
    x_data = in_x_data.flatten()
    y_data = in_y_data.flatten()

    #-- get a mask with those elements posible to compare (non-nan values)
    mask = np.logical_and(np.logical_not(np.isnan(x_data)), np.logical_not(np.isnan(y_data)))
    n_colocations = len(mask[mask==True])
    x_data = x_data[mask]
    y_data = y_data[mask]
    xyerr = np.abs( y_data - x_data )
    sorterr = np.sort(xyerr)
    uncert = sorterr[int(np.size(x_data) * 0.68)]
    uncert_str = "EE = {:.2f}".format(uncert)
    
    Bias   = np.mean( y_data - x_data )
    Bias_str = 'bias = {:4.2f}'.format(Bias)
        
    #-- liner regression
    slope, intercept, correlation, p_value_slope, std_error = stats.linregress(x_data, y_data)
    '''
    if model == 'OLS':
        slope, intercept, correlation, p_value_slope, std_error = stats.linregress(x_data, y_data)
    if model == 'RMA':
        results = regress2(x_data, y_data, _need_intercept=True)

        slope    		= results['slope']
        intercept		= results['intercept']
        correlation		= results['r']
        p_value_slope	= results['pvalue'] 
        std_error		= ''
    '''
    #-- Calculates a Pearson correlation coefficient and the p-value for testing non-correlation
    r, p_value = stats.pearsonr(x_data, y_data)

    rmse   = np.std(y_data-x_data, ddof=0)
    mean_x = np.mean(x_data)
    mean_y = np.mean(y_data)
    std_x  = np.std(x_data, ddof=1)
    std_y  = np.std(y_data, ddof=1)

    paths = ax.scatter(x_data, y_data, s = markersize, c = 'gray', alpha = alpha)
    '''
    #-- create scatter plot
    if stdOn == False:
        paths = ax.scatter(x_data, y_data, s = markersize, c = color, alpha = alpha)

    if stdOn == True:
        paths = ax.errorbar(x_data, y_data, yerr = yerr,  xerr = xerr, c = color, \
                            ecolor = color, fmt=fmt, \
                            markersize = markersize, elinewidth=elinewidth, capsize = capsize)
    '''
    if case_str is not None:
        case_str = case_str + '\n'
    else:
        case_str = ''

    min_x = np.nanmin(x_data)
    max_x = np.nanmax(x_data)
    min_y = np.nanmin(y_data)
    max_y = np.nanmax(y_data)

    # make the statistics...
    if statOn == True:
        #-- add slope line

        x = np.array((min_x- 2 *np.absolute(min_x),max_x+ 2 * np.absolute(max_x)))
        y = (slope * x) + intercept
        if regress_line:
            ax.plot(x, y, '-', color=reg_color, linewidth=1.2)
        if one2one_line:
            ax.plot(x, x, '-', color='k', linewidth=0.8)


        ee_string = ''
        if eeOn == True:
            err_frac = eeFrac
            bias = eeBias
            y1 = x - x * err_frac - bias
            y2 = x + x * err_frac + bias
            if ee_line2 == True:
                ax.plot(x, y1, '--', color = 'k', linewidth=0.8)
                ax.plot(x, y2, '--', color = 'k', linewidth=0.8)
            if ee_line1 == True:
                ax.plot([0+uncert, 10], [0,(1.0)*10-uncert], 'k--',linewidth=0.5)
                ax.plot([0, (10-uncert)/(1.0)], [0+uncert,10], 'k--',zorder=0,linewidth=0.5)  
            num = 0
            for i in range(len(x_data)):
                upp = x_data[i] * (1 + err_frac) + bias
                low = x_data[i] * (1 - err_frac) - bias
                if (y_data[i] > low) & (y_data[i] < upp):	
                    num = num + 1
            ee_static = np.round( num / len(x_data) * 100, 3)

            ee_string = 'EE%: ' + str(ee_static) + '%' + '\n' + '(EE = ' + str(err_frac) + 'AOD+' + str(bias) + ')'
            print(ee_string)

        #-- create strings for equations in the plot
        correlation_string = "R = {:.2f}".format(r)

        sign = " + "
        if intercept < 0:
            sign = " - "

        lineal_eq = "y = " + str(round(slope, 3)) + dp_x + sign + str(round(abs(intercept), 3))+ '\n'
        rmse_coef = "RMSE = " + str(round(rmse,3)) + '\n'

        if rmse_coef_flag == False:
            rmse_coef = ''

        if p_value >= 0.05:
            p_value_s = "(p > 0.05)"
        else:
            if p_value < 0.01:
                p_value_s = "(p < 0.01)"
            else:
                p_value_s = "(p < 0.05)"

        n_collocations = "N = " + str(n_colocations) + '\n' 
        x_mean_std = "x: " + str(round(mean_x, 3)) + " $\pm$ " + str(round(abs(std_x), 3)) + '\n'
        y_mean_std = "y: " + str(round(mean_y, 3)) + " $\pm$ " + str(round(abs(std_y), 3)) + '\n'

        print(correlation_string + ' ' + p_value_s)

        equations0 = case_str + \
                     n_collocations + \
                     x_mean_std + \
                     y_mean_std + \
                     rmse_coef  + \
                     lineal_eq  + \
                     correlation_string + ' ' + p_value_s +  '\n' + \
                     Bias_str
        if ee_line1 == True:
            equations0 = equations0 + '\n' + uncert_str
        if ee_line2 == True:
            equations0 = equations0 + '\n' + ee_string

    if label_p == None:
        if r>0:
            label_p = 'upper left'
        else:
            label_p = 'lower left'

    if (label_p == 'positive'):
        # divided
        equations1 = case_str + \
                     n_collocations + \
                     x_mean_std + \
                     y_mean_std + \
                     rmse_coef + \
                     lineal_eq
        equations2 = correlation_string + ' ' + p_value_s +  '\n' + \
                     Bias_str
        if ee_line1 == True:
            equations2 = equations2 + '\n' + uncert_str
        if ee_line2 == True:
            equations2 = equations2 + '\n' + ee_string


        posXY1      = (0, 1)
        posXY_text1 = (5, -5)
        ax.annotate(equations1, xy=posXY1, xytext=posXY_text1, va='top', \
                xycoords='axes fraction', color=color, textcoords='offset points',fontsize=fsize)


        posXY2      = (1, 0)
        posXY_text2 = (-5, 5)
        ax.annotate(equations2, xy=posXY2, xytext=posXY_text2, va='bottom', ha='right', \
                xycoords='axes fraction', color=color, textcoords='offset points',fontsize=fsize)


    elif (label_p == 'upper left'):
        # upper left
        posXY0      = (0, 1)
        posXY_text0 = (5, -5)
        ax.annotate(equations0, xy=posXY0, xytext=posXY_text0, va='top', \
                xycoords='axes fraction', textcoords='offset points', color=color, fontsize=fsize)

    elif (label_p == 'lower right'):
        # lower right
        posXY0      = (1, 0)
        posXY_text0 = (-5, 5)
        ax.annotate(equations0, xy=posXY0, xytext=posXY_text0, va='bottom', ha='right', \
                xycoords='axes fraction', textcoords='offset points', color=color, fontsize=fsize)


    elif (label_p == 'lower left'):
        # lower right
        posXY0      = (0, 0)
        posXY_text0 = (5, 5)
        ax.annotate(equations0, xy=posXY0, xytext=posXY_text0, va='bottom', ha='left', \
                xycoords='axes fraction', textcoords='offset points', color=color, fontsize=fsize)

    elif (label_p == 'upper right'):
        # upper right
        posXY0      = (1, 1)
        posXY_text0 = (-5,-5)
        ax.annotate(equations0, xy=posXY0, xytext=posXY_text0, va='top', ha='right', \
                xycoords='axes fraction', textcoords='offset points', color=color, fontsize=fsize)

    else:
        print('!!! scatter: label_p error !!!')
        exit()

    if yRange == None:
        yRange 		= []
        yRange.append(min_y)
        yRange.append(max_y)

    if xRange == None:
        xRange 		= []
        xRange.append(min_x)
        xRange.append(max_x)

    if delta_ticks_X == None:
        delta_ticks_X = (xRange[1] - xRange[0])/5
    if delta_ticks_Y == None:
        delta_ticks_Y = (yRange[1] - yRange[0])/5


    ax.set_ylim(yRange[0], yRange[1])
    ax.set_xlim(xRange[0], xRange[1])


    ax.set_ylabel(ylabel, fontsize=fsize)
    ax.set_xlabel(xlabel, fontsize=fsize)	
    ax.tick_params(labelsize=fsize)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_major_locator(MultipleLocator(delta_ticks_X))
    ax.yaxis.set_major_locator(MultipleLocator(delta_ticks_Y))	

    return paths, slope, intercept
