import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
import cv2 as cv


def plot_snake(ax, V, fill='green', line='red', alpha=1, with_txt=False):
    """ plots the snake onto a sub-plot
    :param ax: subplot (fig.add_subplot(abc))
    :param V: point locations ( [ (x0, y0), (x1, y1), ... (xn, yn)]
    :param fill: point color
    :param line: line color
    :param alpha: [0 .. 1]
    :param with_txt: if True plot numbers as well
    :return:
    """
    V_plt = np.append(V.reshape(-1), V[0,:]).reshape((-1, 2))
    ax.plot(V_plt[:,0], V_plt[:,1], color=line, alpha=alpha)
    ax.scatter(V[:,0], V[:,1], color=fill,
               edgecolors='black',
               linewidth=2, s=50, alpha=alpha)
    if with_txt:
        for i, (x, y) in enumerate(V):
            ax.text(x, y, str(i))


def load_data(fpath, radius):
    """
    :param fpath:
    :param radius:
    :return:
    """
    img = cv.imread(fpath, 0)
    h, w = img.shape
    n = 20  # number of points
    u = lambda i: radius * np.cos(i) + w / 2
    v = lambda i: radius * np.sin(i) + h / 2
    V = np.array(
        [(u(i), v(i)) for i in np.linspace(0, 2 * np.pi, n + 1)][0:-1],
        'int32')

    return img, V


# ===========================================
# RUNNING
# ===========================================

# FUNCTIONS
# ------------------------
def parse_state(s):
    x = s % 9
    y = s // 9
    if s in [0,3,6]:
        if s == 0:
            y = -y
        x = -x
    return x,y

# ------------------------


def run(fpath, radius):
    """ run experiment
    :param fpath:
    :param radius:
    :return:
    """
    img, V = load_data(fpath, radius)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    n_steps = 200

    # ------------------------
    n = len(V) # number of points
    img_grad_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
    img_grad_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
    U = -(img_grad_x ** 2 + img_grad_y ** 2)
    
    # as it was not specified in the lecture, we'll just set it to 1
    alpha = 0.001

    # ------------------------

    for t in range(n_steps):
        # ------------------------
        
        #minimize the term: U + P

        #calculate the pairwise costs
        L = []
        for i in range(n-1):
            y,x = V[i]
            w_n0 = img[x-1:x+2, y-1:y+2].reshape(-1,1)
            y,x = V[i+1]
            w_n1 = img[x-1:x+2, y-1:y+2].reshape(-1,1)
            
            L.append(
                alpha * ((w_n0 ** 2) + (w_n1[:, np.newaxis] ** 2)).reshape(-1,1)
            )

        
        #add the unary cost
        for i in range(n-1):
            y,x = V[i]
            u_n = U[x-1:x+2, y-1:y+2].reshape(-1,1)

            L[i] = (L[i] + u_n[:, np.newaxis]).reshape(-1,1)


        # sum up to get the total traversal cost
        s = sum(L)
        y,x = V[-1]
        u_n = U[x-1:x+2, y-1:y+2].reshape(-1,1)
        end = (s + u_n[:, np.newaxis]).reshape(-1,1)
        
        # save y, search onwards with x
        min_pos = np.argmin(end)

        min_states = [int(((min_pos % (9**3)) % (9**2)) // 9)]
        # min_pos = ((min_pos % (9**3)) % (9**2)) % 9
        min_pos = min_states[0]

        for item in L[::-1]:
            s -= item
            start = (9**2 + 9)*min_pos
            end = 9**2 * min_pos + 9 *(min_pos+1)
            min_pos = np.argmin(s[start:end])
            min_states.append(min_pos)

        min_states = min_states[::-1]

        for i, state in enumerate(min_states):
            y,x = parse_state(state)
            V[i,0] = V[i,0] + x
            V[i,1] = V[i,1] + y

        # ------------------------

        ax.clear()
        ax.imshow(img, cmap='gray')
        ax.set_title('frame ' + str(t))
        plot_snake(ax, V)
        plt.pause(0.01)

    plt.pause(2)


if __name__ == '__main__':
    run('images/ball.png', radius=120)
    run('images/coffee.png', radius=100)
