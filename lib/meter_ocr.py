import cv2
import logging
import inspect
import numpy as np


class MeterOcr:
    _log_count = 0

    def __init__(self, path):
        self._path = path

    def getDigits(self):
        img = cv2.imread(self._path, cv2.IMREAD_COLOR)

        center = self.find_sample(img)

        lines = self.get_lines(img, center)

        img = self.align_and_crop_with_lines(img, lines, center)

    def align_and_crop_with_lines(self, img, lines, center):
        def y_for_line(x, r, theta):
            if theta == 0:
                return np.nan
            return (r - (x * np.cos(theta))) / np.sin(theta)

        h, w, k = img.shape
        x_center, y_center = center
        rho_above, theta_above = lines[0][0]
        rho_below, theta_below = lines[1][0]
        #M = cv2.getRotationMatrix2D(
        #    (0, (rho_below-rho_above)/2+rho_above), theta_above/np.pi*180-90, 1)
        
        #img = cv2.warpAffine(img, M, (w, h))

        pts1 = np.float32([
            [x_center - 100, y_for_line(x_center - 100, rho_above, theta_above)],
            [x_center + 100, y_for_line(x_center + 100, rho_above, theta_above)],
            [x_center - 100, y_for_line(x_center - 100, rho_below, theta_below)],
            [x_center + 100, y_for_line(x_center + 100, rho_below, theta_below)],
        ])
        y_above_center = int(y_for_line(x_center, rho_above, theta_above))
        y_below_center = int(y_for_line(x_center, rho_below, theta_below))
        pts2 = np.float32([
            [x_center - 100, y_above_center],
            [x_center + 100, y_above_center],
            [x_center - 100, y_below_center],
            [x_center + 100, y_below_center],
        ])

        img = self.draw_lines(img, lines)
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (w,h))
        
        self.log_image(img)

        img = img[y_above_center:y_below_center, 0:w]
        
        self.log_image(img)
        

    def get_lines(self, img, center):
        h, w, k = img.shape
        x_center, y_center = center
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 100, 300)
        self.log_image(edges, 'edges')

        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=110)
        self.log_image(self.draw_lines(img, lines))

        rho_below = rho_above = np.sqrt(h*h+w*w)
        line_above = None
        line_below = None
        for line in lines:
            rho, theta = line[0]
            sin = np.sin(theta)
            cos = np.cos(theta)
            if (sin < 0.8):
                continue
            rho_center = x_center*cos + y_center*sin
            if rho_center > rho and rho_center-rho < rho_above:
                rho_above = rho_center-rho
                line_above = (rho, theta), (sin, cos)

            if rho_center < rho and rho-rho_center < rho_below:
                rho_below = rho-rho_center
                line_below = (rho, theta), (sin, cos)

        if line_below is None or line_above is None:
            raise Exception("No lines found")

        self.log_image(self.draw_lines(img, [line_above, line_below]))

        return [line_above, line_below]

    def find_sample(self, img):
        sample = cv2.imread("samples/sample.jpg")
        sample_h, sample_w, sample_k = sample.shape
        res = cv2.matchTemplate(img, sample, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        self.log('max_loc', max_loc)

        x_center = max_loc[0] + sample_w/2
        y_center = max_loc[1] + sample_h/2

        self.log('max_loc', {'x_center': x_center, 'y_center': y_center})

        log_img = cv2.rectangle(img.copy(), (max_loc[0], max_loc[1]), (
            max_loc[0] + sample_w, max_loc[1] + sample_h), (0, 0, 255), 5)
        self.log_image(log_img)

        return x_center, y_center

    def log(self, message, data=None):
        log_msg = message
        if data:
            log_msg = '%s: %s' % (log_msg, data)
        logging.debug(log_msg)

    def draw_lines(self, img, lines):
        log_image = img.copy()
        if lines is not None:
            for i in range(0, len(lines)):
                rho, theta = lines[i][0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
                pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
                cv2.line(log_image, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
        return log_image

    def log_image(self, img, step=None, prefix=None):

        if not step:
            step = inspect.stack()[1].frame.f_code.co_name
        if not prefix:
            prefix = f'{self._log_count:02d}'
        self._log_count += 1
        cv2.imwrite(f'result/{prefix}_meter_{step}.jpg', img)
