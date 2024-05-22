def _hooke_best_nearby(f, delta, point, prevbest, args=[]):
    '''
        ## given a point, look for a better one nearby
        one coord at a time
        
        f is a function that takes a list of floats (of the same length as point) as an input
        args is a dict of any additional arguments to pass to f
        delta, and point are same-length lists of floats
        prevbest is a float
        
        point and delta are both modified by the function
    '''
    z = [x for x in point]
    minf = prevbest
    ftmp = 0.0
    
    fev = 0
    
    for i in range(len(point)):
        #see if moving point in the positive delta direction decreases the 
        z[i] = point[i] + delta[i]
        ftmp = f(z, *args)
        fev += 1
        if ftmp < minf:
            minf = ftmp
        else:
            #if not, try moving it in the other direction
            delta[i] = -delta[i]
            z[i] = point[i] + delta[i]
            ftmp = f(z, *args)
            fev += 1
            if ftmp < minf:
                minf = ftmp
            else:
                #if moving the point in both delta directions result in no improvement, then just keep the point where it is
                z[i] = point[i]

    for i in range(len(z)):
        point[i] = z[i]
    return (minf, fev)
                
        
def hooke(f, startpt, rho=0.5, epsilon=1E-6, itermax=5000, args=[]):
    result = dict()
    result['success'] = True
    result['message'] = 'success'
    
    delta = [0.0] * len(startpt)
    xbefore = [x for x in startpt]
    newx = [x for x in startpt]
    endpt = [0.0] * len(startpt)
    
    fmin = None
    nfev = 0
    iters = 0
    
    try:
        for i in range(len(startpt)):
            delta[i] = abs(startpt[i] * rho)
            if (delta[i] == 0.0):
                # we always want a non-zero delta because otherwise we'd just be checking the same point over and over
                # and wouldn't find a minimum
                delta[i] = rho

        steplength = rho

        fbefore = f(newx, *args)
        nfev += 1
        
        newf = fbefore
        fmin = newf
        while ((iters < itermax) and (steplength > epsilon)):
            iters += 1
            #print "after %5d , f(x) = %.4le at" % (funevals, fbefore)
            
    #        for j in range(len(startpt)):
                #print "   x[%2d] = %4le" % (j, xbefore[j])
    #            pass
            
            ##/* find best new point, one coord at a time */
            newx = [x for x in xbefore]
            (newf, evals) = _hooke_best_nearby(f, delta, newx, fbefore, args)
            nfev += evals
            ##/* if we made some improvements, pursue that direction */
            keep = 1
            while ((newf < fbefore) and (keep == 1)):
                fmin = newf
                for i in range(len(startpt)):
                    ##/* firstly, arrange the sign of delta[] */
                    if newx[i] <= xbefore[i]:
                        delta[i] = -abs(delta[i])
                    else:
                        delta[i] = abs(delta[i])
                    ## /* now, move further in this direction */
                    tmp = xbefore[i]
                    xbefore[i] = newx[i]
                    newx[i] = newx[i] + newx[i] - tmp
                fbefore = newf
                (newf, evals) = _hooke_best_nearby(f, delta, newx, fbefore, args)
                nfev += evals
                ##/* if the further (optimistic) move was bad.... */
                if (newf >= fbefore):
                    break
                
                ## /* make sure that the differences between the new */
                ## /* and the old points are due to actual */
                ## /* displacements; beware of roundoff errors that */
                ## /* might cause newf < fbefore */
                keep = 0
                for i in range(len(startpt)):
                    keep = 1
                    if ( abs(newx[i] - xbefore[i]) > (0.5 * abs(delta[i])) ):
                        break
                    else:
                        keep = 0
            if ((steplength >= epsilon) and (newf >= fbefore)):
                steplength = steplength * rho
                delta = [x * rho for x in delta]
        for x in range(len(xbefore)):
            endpt[x] = xbefore[x]
    except Exception as e:
        result['success'] = False
        result['message'] = str(e)
    finally:
        result['nit'] = iters
        result['fevals'] = nfev
        result['fun'] = fmin
        result['x'] = endpt
    
    return result

def f(x):
    [x1,x2] = x
    return 1*x1 - x2**2 + 2*x1*x2**2 + x2**3

start = [0,1.0]
res = hooke(f, start,rho=0.1, itermax=2)
print("Optimal solution:", res["x"])
print("f(x):", res["fun"])