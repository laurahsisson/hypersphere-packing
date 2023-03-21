import numpy as np
import tqdm

PHYSICS_STEPS = 1e4
EPS = 1e-6

def volume(point):
    norm = np.linalg.norm(point)
    if norm > 1:
        return point/norm
    return point

def surface(point):
    return point / np.linalg.norm(point)

def point(dim,normfn):
    return normfn(np.random.uniform(low=-1,high=1,size=dim))

def overlap(p1,p2,r):
    delta = p1-p2
    d = np.linalg.norm(delta)
    return d, delta

def separate(p1,p2,r,normfn):
    d, delta = overlap(p1,p2,r)
    p1x, p2x, = p1, p2
    if d < r:
        p1x = normfn(p1 + (delta/2))
        p2x = normfn(p2 - (delta/2))
    return d, p1x, p2x
    return has_overlap, p1, p2


def can_fit(points,r,normfn):
    for _ in tqdm.tqdm(range(int(PHYSICS_STEPS))):
        had_overlap = False
        for i, p1 in enumerate(points):
            for j, p2 in enumerate(points):
                if i == j:
                    continue
                d, p1x, p2x = separate(p1,p2,r,normfn)
                points[i], points[j] = p1x, p2x
                had_overlap = had_overlap or d < r
        
        if not had_overlap:
            return points, True

    return points, False


def find_max(dim,r,normfn):
    points = []
    fit = True
    while fit:
        points.append(point(dim,normfn))
        points, fit = can_fit(points,r,normfn)
        print(f"Fit = {fit} with a total of {len(points)}.")
    return len(points) - 1


find_max(2,.5,volume)


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


