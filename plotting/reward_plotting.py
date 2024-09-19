import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_square_and_tri(axis, pos):
    rect = patches.Rectangle(pos, 1, 1, linewidth=1, edgecolor='r', facecolor='none')
    axis.add_patch(rect)
    mid_point = [pos[0] + 0.5, pos[1] + 0.5]

    # left
    tri = plt.Polygon([mid_point, [mid_point[0] - 0.5, mid_point[0] + 0.5], [mid_point[0] - 0.5, mid_point[0] - 0.5]],
                      color="red")
    axis.add_patch(tri)

    # right
    tri = plt.Polygon([mid_point, [mid_point[0] + 0.5, mid_point[0] + 0.5], [mid_point[0] + 0.5, mid_point[0] - 0.5]],
                      color="green")
    axis.add_patch(tri)

    # up
    tri = plt.Polygon([mid_point, [mid_point[0] + 0.5, mid_point[0] + 0.5], [mid_point[0] - 0.5, mid_point[0] + 0.5]],
                      color="blue")
    axis.add_patch(tri)

    # down
    tri = plt.Polygon([mid_point, [mid_point[0] + 0.5, mid_point[0] - 0.5], [mid_point[0] - 0.5, mid_point[0] - 0.5]],
                      color="yellow")
    axis.add_patch(tri)
    return None


fig, ax = plt.subplots()

draw_square_and_tri(ax, [0, 0])

plt.show()