import numpy as np
import tqdm
import matplotlib.pyplot as plt

PHYSICS_STEPS = 1e4
EPS = 1e-2

def volume(point,r):
    norm = np.linalg.norm(point)
    outer_edge = point*(1+(r/norm))
    norm = np.linalg.norm(outer_edge)
    if norm > 1:
        ne = outer_edge/norm
        return point - (outer_edge-ne)

    return point

def surface(point,r):
    return point / np.linalg.norm(point)

def point(dim,normfn,r):
    return normfn(np.random.uniform(low=-1,high=1,size=dim),r)

def perturb(point,r,normfn):
    return normfn(point + np.random.uniform(low=-EPS,high=EPS,size=point.shape),r)

def overlap(p1,p2,r):
    delta = p1-p2
    d = np.linalg.norm(delta)
    return d, delta

def separate(p1,p2,r,normfn):
    d, delta = overlap(p1,p2,r)
    p1x, p2x, = p1, p2
    if d < 2*r:
        p1x = normfn(p1 + (delta/2),r)
        p2x = normfn(p2 - (delta/2),r)
    return d, p1x, p2x


def can_fit(points,r,normfn):
    for _ in tqdm.tqdm(range(int(PHYSICS_STEPS))):
        had_overlap = False
        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points):
                if i == j:
                    continue
                d, p1x, p2x = separate(p1,p2,r,normfn)
                points[i], points[j] = p1x, p2x
                had_overlap = had_overlap or d < 2*r
        
        if not had_overlap:
            return points, True
        else:
            for i, p in enumerate(points):
                points[i] = perturb(p,r,normfn)


    return points, False


def find_max(dim,r,normfn):
    points = []
    fit = True
    while fit:
        next_points = points + [point(dim,normfn,r)]
        next_points, fit = can_fit(next_points,r,normfn)
        if fit:
            print(f"Fit {len(next_points)} in d={dim} w/ r={r}.")
            points = next_points

    return points

def chart2d():
    radius = .2
    points = find_max(2,radius,volume)
    plt.figure(figsize=[5, 5])
    ax = plt.axes([0.1, 0.1, 0.8, 0.8], xlim=(-1, 1), ylim=(-1, 1))

    points_whole_ax = 5 * 0.8 * 72    # 1 point = dpi / 72 pixels

    points_radius = radius / 1.0 * points_whole_ax

    x,y = zip(*points)
    ax.scatter(x,y,s=points_radius**2,facecolors='None', edgecolors='0')
    ax.scatter(x,y,facecolors='0', edgecolors='None')
    ax.scatter(0,0,s=points_whole_ax**2,facecolors='None', edgecolors='0')

    for i in range(len(x)):
        ax.annotate(i, (x[i], y[i]))

    plt.show()


def chartdim():
    vs = []
    sf = []
    radius = .2
    for dim in tqdm.tqdm(range(2,15)):
        vs.append(find_max(dim,radius,volume))
        sf.append(find_max(dim,radius,surface))
        print(f"DONE WITH DIM={dim}")

    fig, ax = plt.subplots()
    ax.plot(vs, label="Volume")
    ax.plot(sf, label="Surface")
    plt.show()


chart2d()
# chartdim()
# nf = volume
# dim = 2
# r = .2
# points = [point()]

# count = 1
# points = []
# while True:

#     points.add(point)

# for i in range(100):
#     p = point(3)
#     n = volume(p)
#     if not np.isclose(n,p).all():
#         print(p)
#         print(n)
#         print()


