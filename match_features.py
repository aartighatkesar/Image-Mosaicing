import numpy as np
import cv2
import os


class SiftKpDesc():
    def __init__(self, kp, desc):
        # List of keypoints in (x,y) crd -> N x 2
        self.kp = kp

        # List of Descriptors at keypoints : N x 128
        self.desc = desc


class SiftMatching:

    _BLUE = [255, 0, 0]
    _GREEN = [0, 255, 0]
    _RED = [0, 0, 255]
    _CYAN = [255, 255, 0]

    _line_thickness = 2
    _radius = 5
    _circ_thickness = 2


    def __init__(self, img_1_path, img_2_path, results_fldr='', nfeatures=2000, gamma=0.8):

        fname_1 = os.path.basename(img_1_path)
        fname_2 = os.path.basename(img_2_path)

        if not results_fldr:
            results_fldr = os.path.split(img_1_path)[0]

        self.result_fldr = os.path.join(results_fldr, 'results')

        self.prefix = fname_1.split('.')[0] + '_' + fname_2.split('.')[0]

        if not os.path.exists(self.result_fldr):
            os.makedirs(self.result_fldr)

        self.img_1_bgr = self.read_image(img_1_path)
        self.img_2_bgr = self.read_image(img_2_path)

        self.nfeatures = nfeatures
        self.gamma = gamma


    def read_image(self, img_path):

        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)

        return img_bgr



    def get_sift_features(self, img_bgr, nfeatures=2000):

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        sift_obj = cv2.xfeatures2d.SIFT_create(nfeatures)

        # kp_list_obj is a list of "KeyPoint" objects with location stored as tuple in "pt" attribute
        kp_list_obj, desc = sift_obj.detectAndCompute(image=img_gray, mask=None)

        kp = [x.pt for x in kp_list_obj]

        return SiftKpDesc(kp, desc)


    def match_features(self, sift_kp_desc_obj1, sift_kp_desc_obj2, gamma=0.8):
        correspondence = []  # list of lists of [x1, y1, x2, y2]

        for i in range(len(sift_kp_desc_obj1.kp)):
            sc = np.linalg.norm(sift_kp_desc_obj1.desc[i] - sift_kp_desc_obj2.desc, axis=1)
            idx = np.argsort(sc)

            val = sc[idx[0]] / sc[idx[1]]

            if val <= gamma:
                correspondence.append([*sift_kp_desc_obj1.kp[i], *sift_kp_desc_obj2.kp[idx[0]]])

        return correspondence


    def draw_correspondence(self, correspondence, img_1, img_2):

        if len(img_1.shape) == 2:
            img_1 = np.repeat(img_1[:, :, np.newaxis], 3, axis=2)

        if len(img_2.shape) == 2:
            img_2 = np.repeat(img_2[:, :, np.newaxis], 3, axis=2)

        h, w, _ = img_1.shape

        img_stack = np.hstack((img_1, img_2))

        for x1, y1, x2, y2 in correspondence:
            x1_d = int(round(x1))
            y1_d = int(round(y1))

            x2_d = int(round(x2) + w)
            y2_d = int(round(y2))

            cv2.circle(img_stack, (x1_d, y1_d), radius=self._radius, color=self._BLUE,
                       thickness=self._circ_thickness, lineType=cv2.LINE_AA)

            cv2.circle(img_stack, (x2_d, y2_d), radius=self._radius, color=self._BLUE,
                       thickness=self._circ_thickness, lineType=cv2.LINE_AA)

            cv2.line(img_stack, (x1_d, y1_d), (x2_d, y2_d), color=self._CYAN,
                     thickness=self._line_thickness)

        fname = os.path.join(self.result_fldr, self.prefix + '_sift_corr.jpg')
        cv2.imwrite(fname, img_stack)


    def run(self):

        sift_kp_desc_obj1 = self.get_sift_features(self.img_1_bgr, nfeatures=self.nfeatures)
        sift_kp_desc_obj2 = self.get_sift_features(self.img_2_bgr, nfeatures=self.nfeatures)

        correspondence = self.match_features(sift_kp_desc_obj1, sift_kp_desc_obj2, gamma=self.gamma)

        self.draw_correspondence(correspondence, self.img_1_bgr, self.img_2_bgr)

        return correspondence



if __name__ == "__main__":
    img_1_path = "/Users/aartighatkesar/Documents/Image-Mosaicing/input/p3/4.jpg"
    img_2_path = "/Users/aartighatkesar/Documents/Image-Mosaicing/input/p3/5.jpg"

    siftmatch_obj = SiftMatching(img_1_path, img_2_path, results_fldr='', nfeatures=2000, gamma=0.6)
    siftmatch_obj.run()






