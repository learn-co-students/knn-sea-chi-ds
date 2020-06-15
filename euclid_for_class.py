def euclid(point_1, point_2):
    
    # determine how to subtract x, y, z positions
    x = point_1[0] - point_2[0]
    y = point_1[1] - point_2[1]
    z = point_1[2] - point_2[2]
        
    # square the differences
    x_sq = x**2
    y_sq = y**2
    z_sq = z**2
    # sum the squared differences
    sum_sq = x_sq + y_sq + z_sq
    # take the square root of the sum
    return np.sqrt(sum_sq)

distance = [euclid(new_point, point) for point in points]

sorted_distance = sorted(enumerate(distance), key=lambda x: x[1]) 
sorted_distance_ind = [x[0] for x in sorted_distance]
labels[sorted_distance_ind]