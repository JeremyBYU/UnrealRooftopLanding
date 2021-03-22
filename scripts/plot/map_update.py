import matplotlib.pyplot as plt
import matplotlib
from descartes import PolygonPatch
import numpy as np
from shapely.geometry import box, Polygon, Point
import bezier

def plot_geom(ax, geom, x_lim=[-2, 12], y_lim=[-2, 12], shell_color=[]):

    polygon_patch = PolygonPatch(Polygon(geom.exterior))
    ax.add_patch(polygon_patch)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.xlabel("X")
    ax.ylabel("Y", labelpad=-10)


def plot_polygons(ax, poly,  linewidth=2, shell_color='green', hole_color='orange', flip_xy=False,
                  x_lim=[-2, 12], y_lim=[-2, 12], **kwargs):
    shell_coords = np.array(poly.exterior)
    if flip_xy:
        shell_coords[:, [1, 0]] = shell_coords[:, [0, 1]]
    outline = Polygon(shell=shell_coords)
    outlinePatch = PolygonPatch(
        outline, ec=shell_color, fill=False, linewidth=linewidth, **kwargs)
    ax.add_patch(outlinePatch)

    for hole_poly in poly.interiors:
        shell_coords = np.array(hole_poly)
        if flip_xy:
            shell_coords[:, [1, 0]] = shell_coords[:, [0, 1]]
        outline = Polygon(shell=shell_coords)
        outlinePatch = PolygonPatch(
            outline, ec=hole_color, fill=False, linewidth=linewidth)
        ax.add_patch(outlinePatch)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel("X", labelpad=-3)
    ax.set_ylabel("Y", labelpad=-10)


def create_bezier_curve(nodes, extras):
    new_nodes = np.asfortranarray(nodes).transpose()
    curve = bezier.Curve(new_nodes, degree=2)
    s_vals = np.linspace(0.0, 1.0, 10)
    points = curve.evaluate_multi(s_vals)
    points = np.ascontiguousarray(points).transpose()
    points = np.vstack((points, extras))

    return Polygon(points)


def main():
    big_box = box(0, 0, 10, 10)
    small_box = box(2, 2, 8, 8)
    small_hole = Point(5, 6.2).buffer(0.3)
    other_small_hole = Point(5,7.0).buffer(0.5)
    other_obstruction = box(0, 0, 3, 3)

    big_box_with_hole = big_box.difference(small_hole)
    small_box_with_hole = small_box.difference(other_small_hole)


    bump = 1.0
    nodes_top = np.array([[2.0, 8.0 + bump],[5.0, 8.0], [8.0, 8.0 + bump]])
    node_top_poly = create_bezier_curve(nodes_top, np.array([[8, 8.0], [2, 8.0], [2.0, 8.0 + bump]]))
    nodes_bottom = np.array([[2.0, 2.0 - bump],[5.0, 2.0], [8.0, 2.0 - bump]])
    node_bottom_poly = create_bezier_curve(nodes_bottom, np.array([[8, 2.0], [2.0, 2.0], [2.0, 2.0 - bump]]))
    small_box_with_hole_lidar =small_box_with_hole.union(node_top_poly).union(node_bottom_poly).difference(other_obstruction)


    # intersection_poly = small_box_with_hole_lidar.intersection(small_box).intersection(big_box_with_hole)
    intersection_poly = small_box_with_hole_lidar.intersection(big_box_with_hole)
    updated_poly = intersection_poly.union(big_box_with_hole.difference(small_box))


    font = {'size': 14}
    matplotlib.rc('font', **font)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(5, 5))
    plot_polygons(ax[0][0], big_box_with_hole)
    # plot_polygons(ax[0][0], small_box, shell_color='red')

    plot_polygons(ax[0][1], small_box_with_hole_lidar)
    plot_polygons(ax[0][1], small_box, shell_color='red')
    plot_polygons(ax[1][0], intersection_poly)
    plot_polygons(ax[1][1], updated_poly)
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    plot_polygons(ax, big_box_with_hole)
    fig.savefig('assets/imgs/map_update_a.pdf')
    ax.clear()

    plot_polygons(ax, small_box_with_hole_lidar)
    fig.savefig('assets/imgs/map_update_b.pdf')
    ax.clear()

    plot_polygons(ax, intersection_poly)
    plot_polygons(ax, small_box, shell_color='red')
    fig.savefig('assets/imgs/map_update_c.pdf')
    ax.clear()

    plot_polygons(ax, updated_poly)
    fig.savefig('assets/imgs/map_update_d.pdf')
    ax.clear()





if __name__ == '__main__':
    main()
