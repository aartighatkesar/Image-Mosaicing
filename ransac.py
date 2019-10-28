from estimate_homography import *
import numpy as np


class RANSAC:
    _BLUE = [255, 0, 0]
    _GREEN = [0, 255, 0]
    _RED = [0, 0, 255]
    _CYAN = [255, 255, 0]

    _line_thickness = 2
    _radius = 5
    _circ_thickness = 2

    def __init__(self, p=0.99, eps=0.6, n=6, delta=3):
        """
        :param correspondence: np array of [x1, y1, x2, y2] rows where (x1,y1) in img1 matches with (x2,y2) in img2
        :param p: probability that at least one of the N trials will be free of outliers
        :param eps: probability that a given datapoint is an outlier
        :param n: No of datapoints to be sampled per trial
        :param delta: Threshold to determine if a given datapoint is an inlier or outlier
        """
        self.n = n
        self.p = p
        self.eps = eps
        self.delta = delta
        self.N = self.compute_N(self.n, self.p, self.eps)


    def compute_N(self, n, p, eps):

        #N = ln(1-p)/ln(1-(1-eps)^n)
        # N is the number of trials such that atleast one trial is free of outliers

        N = np.round(np.log(1-p)/np.log(1-(1-eps)**n))
        return N


    def sample_n_datapts(self, n_total, n=6):
        """

        :param ntotal: total number of points
        :param n: no of sample correspondences to pick
        :return:
        """
        # id's indicating sampled n points
        idx = np.random.choice(n_total, n, replace=False)

        # id's of points not sampled
        n_idx = np.setdiff1d(np.arange(n_total), idx)

        return idx, n_idx


    def get_inliers(self, H, pts_in_expected, delta):
        """

        :param H: Transformation matrix expected_pts = H * in_pts
        :param pts_in_expected: rows of [x1, y1, x2, y2]
        :param delta: threshold between expected and computed points
        :return:
        """

        pts_in = pts_in_expected[:, 0:2]
        pts_expected = pts_in_expected[:, 2:]

        pts_in = convert_to_homogenous_crd(pts_in, axis=1)  # rows of [x1, y1, 1]

        est_pts = np.matmul(H, pts_in.T)  # cols of [x1, y1, z1]  est_pts = H*in_pts

        est_pts = est_pts/est_pts[-1, :]
        est_pts = est_pts.T  # Rows of [x1, y1, 1]

        dst = np.linalg.norm(est_pts[:, 0:2] - pts_expected, axis=1)

        inliers = pts_in_expected[np.where(dst <= delta)]

        outliers = pts_in_expected[np.where(dst > delta)]

        return inliers, outliers



    def run_ransac(self, correspondence):
        """

        :param correspondence: list of lists/ nd array of rows of x1, y1(in img1), x2, y2(in img2) where x2, y2 = H * x1, y1
        :return current_inliers_cnt : number of inlier correspondence
        :return  current_inliers : ndarray of [x1, y1 x2, y2] rows
        :return current_outliers : ndarray of [x1, y1 x2, y2] rows
        :return current_sample_pts: ndarray of [x1, y1 x2, y2] rows -> sample correspondence chosen during RANSAC
        :return final_H: 3x3 nd array of transformation matrix computed using inliers and sample points. x2, y2 = final_H * x1, y1
        """

        if isinstance(correspondence, list):
            correspondence = np.array(correspondence)

        # # Minimum number of inliers to be accepted as valid set
        n_total = correspondence.shape[0]
        self.M = (1-self.eps)*n_total

        print("N: {}, n: {}, M:{}, p: {}, eps: {}, delta: {}".format(self.N, self.n, self.M,
                                                                     self.p, self.eps, self.delta))
        no_iter = 0

        current_inliers = []
        current_inliers_cnt = 0

        current_sample_pts = []
        current_outliers = []

        while no_iter <= self.N:


            idx, n_idx = self.sample_n_datapts(n_total, self.n)

            sample_pts = correspondence[idx]
            other_pts = correspondence[n_idx]

            # in_pts = H*out_pts
            H = calculate_homography(in_pts=sample_pts[:, 2:], out_pts=sample_pts[:, 0:2])

            inliers, outliers = self.get_inliers(H, other_pts, delta=self.delta)

            inlier_count = inliers.shape[0]

            print("prev_inlier_cnt: {}, new_inlier_cnt: {}".
                  format(current_inliers_cnt, inlier_count))

            if (inlier_count > self.M) and (inlier_count > current_inliers_cnt):
                print(" #### Found better sample of points. Updating #####")
                current_inliers = inliers
                current_outliers = outliers
                current_inliers_cnt = inlier_count
                current_sample_pts = sample_pts

            print(" Done {}/{}".format(no_iter, self.N))

            no_iter += 1

        final_corr_points = np.concatenate((current_sample_pts, current_inliers), axis=0)
        final_H = calculate_homography(in_pts=final_corr_points[:, 2:],
                                                    out_pts=final_corr_points[:, 0:2])

        return current_inliers_cnt, current_inliers, current_outliers, current_sample_pts, final_H


    def draw_lines(self, corr_pts, img_1, img_2, save_path, line_color, pt_color):
        """
        Function to draw lines to indicate correspondence
        :param corr_pts: nd array of points from ing_1 to img2 [x1, y1, x2, y2] rows
        :param img_1: RGB ndarray for image 1
        :param img_2: RGB ndarray for image 2
        :param save_path: Full path to save result image
        :param line_color: color of line. 3 tuple RGB
        :param pt_color: color of point marking coorresponding points, 3 tuple of RGB
        :return:
        """

        h, w, _ = img_1.shape

        img_stack = np.hstack((img_1, img_2))

        for x1, y1, x2, y2 in corr_pts:
            x1_d = int(round(x1))
            y1_d = int(round(y1))

            x2_d = int(round(x2) + w)
            y2_d = int(round(y2))

            cv2.circle(img_stack, (x1_d, y1_d), radius=self._radius, color=pt_color,
                       thickness=self._circ_thickness, lineType=cv2.LINE_AA)

            cv2.circle(img_stack, (x2_d, y2_d), radius=self._radius, color=pt_color,
                       thickness=self._circ_thickness, lineType=cv2.LINE_AA)

            cv2.line(img_stack, (x1_d, y1_d), (x2_d, y2_d), color=line_color,
                     thickness=self._line_thickness)

        cv2.imwrite(save_path, img_stack)




