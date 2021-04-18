from Display import Display
from Grid import Grid
import matplotlib.pyplot as plt
cell_colors = {
            0: '#FFFFFF',
            2: '#EEE4DA',
            4: '#ECE0C8',
            8: '#ECB280',
            16: '#EC8D53',
            32: '#F57C5F',
            64: '#E95937',
            128: '#F3D96B',
            256: '#F2D04A',
            512: '#E5BF2E',
            1024: '#E2B814',
            2048: '#EBC502',
            4096: '#00A2D8',
            8192: '#9ED682',
            16384: '#9ED682',
            32768: '#9ED682',
            65536: '#9ED682',
            131072: '#9ED682',
        }
class MatplotlibDisplay(Display):
    def __init__(self):


        self.ncols = 4
        self.nrows = 4

        self.fig = plt.figure(figsize=(3, 3))
        plt.suptitle('current grid')
        nrows = self.nrows
        ncols = self.ncols
        self.axes = [self.fig.add_subplot(nrows, ncols, r * ncols + c) for r in range(0, nrows) for c in range(1, ncols + 1)]

    def display(self,grid):

        for i, ax in enumerate(self.axes):
            ax.clear()
            x, y = i // self.nrows, i % self.nrows
            ax.text(0.5, 0.5, str(int(grid.mat[x][y])), horizontalalignment='center', verticalalignment='center')
            ax.set_facecolor(cell_colors[int(grid.mat[x][y])])
            ax.set_xticks([])
            ax.set_yticks([])

        plt.pause(0.1)


if __name__ == '__main__':
    g = Grid()
    g.mat[0][0] = 2
    g.mat[1][0] = 2
    g.mat[3][0] = 4

    md = MatplotlibDisplay()
    md.display(g)

