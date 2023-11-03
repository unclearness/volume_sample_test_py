import numpy as np
import obj_io
from scipy.spatial import KDTree


def get_nearest_normal(points, normals, querys):
    tree = KDTree(points)
    _, idx = tree.query(querys, k=1)
    idx = np.array(idx, dtype=int)
    return normals[idx.flatten()].reshape(-1, 3), \
        points[idx.flatten()].reshape(-1, 3)


def filter_outside_points(points, normals, querys):
    nn_normals, nn_points = get_nearest_normal(points, normals, querys)
    nn_vec = nn_points - querys
    nn_vec = nn_vec / np.linalg.norm(nn_vec, axis=-1)[..., None]
    dot = np.sum(nn_normals * nn_vec, axis=-1)
    return querys[dot > 0]


def interpolate(points, attrs, querys, K, eps=1e-10):
    tree = KDTree(points)
    d, idx = tree.query(querys, k=K)
    idx = np.array(idx, dtype=int)
    inv_d = np.power(1.0 / (d + eps), 2.0)
    sum_invd = np.sum(inv_d, axis=-1, keepdims=True)
    interp_coeffs = inv_d / sum_invd
    nn = attrs[idx.flatten()].reshape(-1, K, 3)
    return np.sum(nn * interp_coeffs[..., None], axis=1)


mesh = obj_io.loadObj("suzanne2.obj")
mesh.recomputeNormals()
mesh.vert_colors = (mesh.verts.copy() + 1.5) * 0.5 / 1.5
obj_io.saveObj("suzanne_col.obj", mesh)


x = np.linspace(-1.5, 1.5, 30)
xv, yv, zv = np.meshgrid(x, x, x)
v = np.stack([xv, yv, zv], axis=-1)

verts = v.reshape(-1, 3)
vert_colors = (v.reshape(-1, 3) + 1.5) * 0.5 / 1.5
obj_io.saveObjSimple("vol.obj", verts, [],
                     vert_colors=vert_colors)


q_poss = np.random.rand(999999, 3) * 1.5 * 2 - 1.5
q_poss = filter_outside_points(mesh.verts, mesh.normals /
                               np.linalg.norm(mesh.normals, axis=-1)[..., None], q_poss)

obj_io.saveObjSimple("sample.obj", q_poss, [],
                     vert_colors=interpolate(verts, vert_colors, q_poss, 8))
