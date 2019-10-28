import numpy as np
from estimate_homography import convert_to_homogenous_crd

class OptimizeResult():
    # Result object from an optimization method
    def __init__(self, x=0, nint=0, success=True, message='', min_cost=0):
        """
        :param x: ndarray : The solution of the optimization
        :param nint: int : Number of iterations performed by the optimizer.
        :param success: bool: Whether or not the optimizer exited successfully.
        :param message: str : Description of the cause of the termination
        :param min_cost: float : Cost evaluated at x (solution to optimization)
        """
        self.x = x
        self.nint = nint
        self.success = success
        self.message = message
        self.min_cost = min_cost

    def __repr__(self):
        out = "######## \n solution x: {}\n No of iterations : {} \n Success: {} \n Message : {} \n Min Cost: {} \n ########".\
            format(self.x, self.nint, self.success, self.message, self.min_cost)

        return out


class OptimizeFunction:
    """
    Class with optimization function suite. Returns an output object of "OptimizeResult" class
    """
    def __init__(self, fun, x0, jac, args=()):
        """
        :param fun: function handle to vector to minimize in Non Linear least square sense. Takes input and args
        :param x0: initial guess
        :param jac: Function to return jacobian of fun. Takes same inputs as fun
        :param args: additional args to be passed to fun and jacobian
        :param kwargs: keyword args to be passed to the methods. kwargs such as thresholds etc
        :return result: OptimizeResult class object
        """

        self.result = OptimizeResult(x=x0, nint=0, success=True, message="Initialization", min_cost=0)
        self.x0 = x0
        self.args = args
        self.fun = fun
        self.jac = jac


    def levenberg_marquardt(self, delta_thresh=10**-16, tau=0.5):

        init_jac_f = self.jac(self.x0, *self.args)

        # Compute mu_0
        mu_k = tau * np.amax(np.diag(init_jac_f))

        # initialize
        xk = self.x0

        # iteration counter
        iter = 0

        # update_iter
        update_iter = 0

        # Compute initial residual, cost and result object to reflect initialization
        residual_k = self.fun(xk, *self.args)
        cost_k = np.dot(residual_k.T, residual_k)

        self.result.update_iter = update_iter
        self.result.min_cost = cost_k

        while True:

            # Compute Jacobian of residual
            jac_f = self.jac(xk, *self.args)

            # Compute next delta
            delta_k = np.dot(jac_f.T, jac_f) + mu_k * np.eye(jac_f.shape[1], jac_f.shape[1])  #[Jf_T * Jf + mu*I]
            delta_k = np.linalg.inv(delta_k)  #[Jf_T * Jf + mu*I]^-1
            delta_k = np.dot(delta_k, -1*jac_f.T)
            delta_k = np.dot(delta_k, residual_k)

            # If next update step is less than the threshold, then return
            if np.linalg.norm(delta_k) < delta_thresh or (update_iter > 100):
            # if update_iter > 50:
                self.result.x = xk
                self.result.nint = iter
                self.result.update_iter = update_iter
                self.result.message = '||Delta_k|| < {}'.format(delta_thresh)
                self.result.success = True
                self.result.min_cost = cost_k
                return self.result

            # Compute xk+1
            xk_1 = xk + delta_k

            # Compute eps at xk+1
            residual_k_1 = self.fun(xk_1, *self.args)

            # Compute cost at xk+1
            cost_k_1 = np.dot(residual_k_1.T, residual_k_1)

            # Calculate rho_LM
            num = (cost_k - cost_k_1)
            den = np.dot(np.dot(delta_k.T, -1*jac_f.T), residual_k)
            den = den + np.dot(np.dot(delta_k.T, mu_k * np.eye(jac_f.shape[1], jac_f.shape[1])), delta_k)
            rho_LM = num/den

            # compute mu_k+1
            mu_k = mu_k * max(1/3, 1 - (2 * rho_LM - 1)**3)


            # Update xk to xk+1 only if cost reduces
            if cost_k_1 < cost_k:
                # print("cost_k:{}, cost_k+1: {}".format(cost_k, cost_k_1))

                xk = xk_1
                update_iter += 1
                residual_k = residual_k_1
                cost_k = cost_k_1


            iter += 1


    def dogleg(self):
        pass

    def gauss_newton(self):
        pass

    def gradient_descent(self):
        pass



######## Function for "fun" and "jac" solely for Homography #######
### Use as sample to use LM optimization  ####

def fun_LM_homography(h, x, x_dash):
    """
    Function to pass to OptimizeFunction
    :param h: Vector to be optimized
    :param x: physical coordinates - ndarray of rows of [x1, y1] such that x_dash(in homoogenous crd) = H * x(in homoogenous crd)
    :param x_dash: physical cooordinates - ndarray of rows of x1 y1
    :return:
    """

    H = np.reshape(h, (3,3))

    x_tild = convert_to_homogenous_crd(x, axis=1)  #rows of [x1, y1, 1]
    x_tild = np.dot(H, x_tild.T)
    x_tild = x_tild/x_tild[-1, :]
    x_tild = x_tild.T  # rows of x, y, 1
    x_tild = x_tild[:, 0:2]

    residual = x_dash.flatten() - x_tild.flatten()  # [x`1, y`1, x`2, y`2] - [f11, f21, f12, f22] -> to be optimized by Least squares

    return residual


def jac_LM_homography(h, x, x_dash):

    def jac_fun1(inp_x, inp_h):
        # h = [h11, h12, h13, h21, h22, h23, h31, h32, h33]
        # x = [x1,y1]
        num = inp_h[0] * inp_x[0] + inp_h[1] * inp_x[1] + inp_h[2]
        den = inp_h[6] * inp_x[0] + inp_h[7] * inp_x[1] + inp_h[8]

        # computing deps/dh11 ....deps/dh33
        out = np.zeros_like(inp_h)
        out[0] = -1 * inp_x[0]/den  # deps/dh11
        out[1] = -1 * inp_x[1]/den  #deps/dh12
        out[2] = -1/den  # deps/dh13
        out[6] = (num * inp_x[0])/(den**2)  # deps/dh31
        out[7] = (num * inp_x[1])/(den**2)  #deps/dh32
        out[8] = num/(den**2)  #deps/dh33

        return out

    def jac_fun2(inp_x, inp_h):
        # h = [h11, h12, h13, h21, h22, h23, h31, h32, h33]
        # x = [x1,y1]
        num = inp_h[3] * inp_x[0] + inp_h[4] * inp_x[1] + inp_h[5]
        den = inp_h[6] * inp_x[0] + inp_h[7] * inp_x[1] + inp_h[8]

        # computing deps/dh11 ....deps/dh33
        out = np.zeros_like(inp_h)
        out[3] = -1 * inp_x[0] / den  # deps/dh11
        out[4] = -1 * inp_x[1] / den  # deps/dh12
        out[5] = -1 / den  # deps/dh13
        out[6] = (num * inp_x[0])/(den ** 2)  # deps/dh31
        out[7] = (num * inp_x[1])/(den ** 2)  # deps/dh32
        out[8] = num/(den ** 2)  # deps/dh33

        return out

    jac_eps_1 = np.apply_along_axis(jac_fun1, 1, x, h)
    jac_eps_2 = np.apply_along_axis(jac_fun2, 1, x, h)


    jac_out = np.empty((jac_eps_1.shape[0] + jac_eps_2.shape[0], jac_eps_1.shape[1]))
    jac_out[0::2] = jac_eps_1
    jac_out[1::2] = jac_eps_2

    return jac_out


def func(x):
    return np.array([x[0] + 0.5 * (x[0] - x[1])**3 - 1.0,
            0.5 * (x[1] - x[0])**3 + x[1]])

def jac(x):
    return np.array([[1 + 1.5 * (x[0] - x[1])**2,
                      -1.5 * (x[0] - x[1])**2],
                     [-1.5 * (x[1] - x[0])**2,
                      1 + 1.5 * (x[1] - x[0])**2]])

if __name__ == "__main__":
    from scipy import optimize
    sol = optimize.least_squares(func, [0, 0], jac=jac, method='lm')
    print(sol)
    print("-----")


    opt_obj = OptimizeFunction(fun=func, x0=np.array([0,0]), jac=jac)
    LM_sol = opt_obj.levenberg_marquardt(delta_thresh=1e-6, tau=0.8)
    print(LM_sol)

    # Testing if the solution from scipy and this function are the same
    x_img1 = np.random.randint(20, 50, size=(20, 2))
    H = np.arange(1, 10).reshape(3,3)

    x_temp = convert_to_homogenous_crd(x_img1, axis=1)
    x_tild = np.dot(H, x_temp.T)
    x_tild = x_tild / x_tild[-1, :]
    x_tild = x_tild.T
    x_img2 = x_tild[:, 0:2]

    x_inp = np.concatenate((x_img1, x_img2), axis=1)

    H_noise = H + np.random.randn(3, 3) + 10

    opt_obj = OptimizeFunction(fun=fun_LM_homography, x0=H_noise.flatten(), jac=jac_LM_homography,
                               args=(x_inp[:, 0:2], x_inp[:, 2:]))
    LM_sol = opt_obj.levenberg_marquardt(delta_thresh=1e-24, tau=0.8)

    print("LM_sol : {}, \n &&&& {} &&&&".format(LM_sol, LM_sol.x/LM_sol.x[-1]))

    print("=========")

    sol = optimize.least_squares(fun_LM_homography, H_noise.flatten(), args=(x_inp[:, 0:2], x_inp[:, 2:]), method='lm',
                                 jac=jac_LM_homography,
                                 xtol=1e-24, ftol=1e-24)
    print("scipy sol: \n &&& {}  &&&&".format(sol, sol.x/sol.x[-1]))

    print("=========")

    print("Original : {}".format(H/H[-1, -1]))














