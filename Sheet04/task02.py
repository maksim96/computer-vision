import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import rc


# rc('text', usetex=True)  # if you do not have latex installed simply uncomment this line + line 75


def load_data():
    """ loads the data for this task
    :return:
    """
    fpath = 'images/ball.png'
    radius = 70
    Im = cv2.imread(fpath, 0).astype('float32') / 255  # 0 .. 1

    # we resize the image to speed-up the level set method
    Im = cv2.resize(Im, dsize=(0, 0), fx=0.5, fy=0.5)

    height, width = Im.shape

    centre = (width // 2, height // 2)
    Y, X = np.ogrid[:height, :width]
    phi = radius - np.sqrt((X - centre[0]) ** 2 + (Y - centre[1]) ** 2)

    return Im, phi


def border_to_zero(x):
    x[0] = 0
    x[-1] = 0
    x[:, 0] = 0
    x[:, -1] = 0


def get_contour(phi):
        """ get all points on the contour
        :param phi:
        :return: [(x, y), (x, y), ....]  points on contour
        """
        eps = 1
        A = (phi > -eps) * 1
        B = (phi < eps) * 1
        D = (A - B).astype(np.int32)
        D = (D == 0) * 1
        Y, X = np.nonzero(D)
        return np.array([X, Y]).transpose()

    # ===========================================
    # RUNNING
    # ===========================================

    # FUNCTIONS
    # ------------------------
    # your implementation here

    # ------------------------

if __name__ == '__main__':

    n_steps = 20000
    plot_every_n_step = 100

    Im, phi = load_data()

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # ------------------------
    # your implementation here

    # ------------------------

    I_x = 1 / 2 * (np.roll(Im, -1, axis=1) - np.roll(Im, 1, axis=1))
    I_y = 1 / 2 * (np.roll(Im, -1, axis=0) - np.roll(Im, 1, axis=0))
    w = I_x ** 2 + I_y ** 2  # |gradient(I)|^2
    w = 1 / (w + 1)  # geodesic active contour from task#
    w_x = 1 / 2 * (np.roll(w, -1, axis=1) - np.roll(Im, 1, axis=1)) 
    w_y = 1 / 2 * (np.roll(w, -1, axis=0) - np.roll(Im, 1, axis=0)) 
    # from slide 75
    tau = 0.1 / (4 * np.max(w))
    eps = 10 ^ (-4)


    #there seem to be a bug in the computation
    #del_t_phi_2 behaves reasonably
    #but del_t_phi let the phi function behave chaotically
    for t in range(n_steps):

        phi_x = 1 / 2 * (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1))
        phi_y = 1 / 2 * (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0))

        phi_xx = np.roll(phi, -1, axis=1) + np.roll(phi, 1, axis=1) - 2 * phi
        phi_yy = np.roll(phi, -1, axis=0) + np.roll(phi, 1, axis=0) - 2 * phi

        phi_xy = 1/4*(np.roll(np.roll(phi, -1, axis=1), -1, axis=0) - \
                 np.roll(np.roll(phi, 1, axis=1), -1, axis=0) - \
                 np.roll(np.roll(phi, -1, axis=1), 1, axis=0) + \
                 np.roll(np.roll(phi, 1, axis=1), 1, axis=0))

        border_to_zero(phi_x)
        border_to_zero(phi_y)
        border_to_zero(phi_xx)
        border_to_zero(phi_yy)
        border_to_zero(phi_xy)

        del_t_phi = (phi_xx*np.square(phi_y) - 2*phi_x*phi_y*phi_xy + phi_yy*np.square(phi_x))/(np.square(phi_x) + np.square(phi_y) + eps)

        del_t_phi_2 = np.maximum(w_x, 0) * (np.roll(phi, -1, axis=1) - phi)
        + np.minimum(w_x, 0) * (phi - np.roll(phi, 1, axis=1))
        + np.maximum(w_y, 0) * (np.roll(phi, 1, axis=0) - phi)
        + np.minimum(w_y, 0) * (phi - np.roll(phi, -1, axis=0))

        phi += tau * (w * del_t_phi + del_t_phi_2)

        if t % plot_every_n_step == 0:
            ax1.clear()
            ax1.imshow(Im, cmap='gray')
            ax1.set_title('frame ' + str(t))

            contour = get_contour(phi)
            if len(contour) > 0:
                ax1.scatter(contour[:, 0], contour[:, 1], color='red', s=1)

            ax2.clear()
            ax2.imshow(phi)
            ax2.set_title(r'$\phi$', fontsize=22)
            plt.pause(0.01)

    plt.show()
