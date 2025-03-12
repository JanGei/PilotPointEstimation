from shapely.geometry import LineString, MultiPoint
import flopy

def intersect_with_grid(gwf, coordinates: list, coordinate_type: str = 'MultiPoint') -> list:
    
    modelgrid = gwf.modelgrid
    ixs    = flopy.utils.GridIntersect(modelgrid, method = "vertex")
    
    if coordinate_type == 'MultiPoint':
        intersection   = ixs.intersect(MultiPoint(coordinates))
    elif coordinate_type == 'LineString':
        intersection   = ixs.intersect(LineString(coordinates))
    else:
        print('The coordinate type you provided was not recognized')
        
    
    return intersection.cellids