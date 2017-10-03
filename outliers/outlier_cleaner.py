#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    errors = [ (p-nw) * (p-nw) for p, nw in zip( predictions, net_worths ) ]
    cleaned_data = [ (a, nw, e) for e, a, nw in sorted( zip( errors, ages, net_worths ), key=lambda( e, a, nw ): e ) ]

    numRemainingData = int( round( len(cleaned_data)*0.9, 0) )
    print cleaned_data[0], cleaned_data[-1], len(cleaned_data)
    cleaned_data = cleaned_data[:numRemainingData ]
    print cleaned_data[0], cleaned_data[-1], numRemainingData
    return cleaned_data

