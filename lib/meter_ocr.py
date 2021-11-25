import cv2
import logging
import inspect
import numpy as np


class MeterOcr:
    _log_count = 0
    _ratio = .25

    def __init__(self, path):
        self._path = path

    def getDigits(self):
        img = self.get_image()        

        center = self.find_sample(img, 'samples/sample.jpg')

        img = self.crop_image(img, center)

        center = self.find_sample(img, 'samples/sample.jpg')

        lines = self.get_lines(img, center, 100)

        img = self.align_and_crop_with_lines(img, lines, center)

        center = self.find_sample(img, 'samples/sample.jpg')

        lines = self.get_lines(img, center, 70, 90)

        img = self.align_and_crop_with_lines(img, lines, center)

        img = self.crop_sides(img)

        digits = self.split_digits(img, 7)

        contours = self.get_contours(digits)

    def get_contours(self, digits):
        for digit in digits:

            digit = self.cut_white_sides(digit, (4,4), True)

           
            digit_grey = cv2.cvtColor(digit,cv2.COLOR_BGR2GRAY) 
            digit_grey = cv2.adaptiveThreshold(digit_grey,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 0)  
            self.log_image(digit_grey)
            dh, dw = digit_grey.shape
            contours, hierarhy = cv2.findContours(digit_grey.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            biggest_contour = None
            biggest_contour_area = 0
            for contour in contours:
                M = cv2.moments(contour)

                
                if cv2.contourArea(contour)<30:
                    continue
                
                if cv2.arcLength(contour,True)<30:
                    continue

                
                cx = M['m10']/M['m00']
                cy = M['m01']/M['m00']

                # if cx/dw<0.3 or cx/dw>0.7:
                #     continue
                if cv2.contourArea(contour)>biggest_contour_area:
                    biggest_contour = contour
                    biggest_contour_area = cv2.contourArea(contour)
                    biggest_contour_cx = cx
                    biggest_contour_cy = cy
            self.log_image(cv2.drawContours( digit.copy(), [biggest_contour], -1, (0,0,255), 1, cv2.LINE_8))

            mask = np.zeros(digit.shape,np.uint8)
            cv2.drawContours(mask,[biggest_contour],0,255,-1)
            digit = cv2.bitwise_and(digit,digit,mask = mask)
            self.log_image(digit)    

    def split_digits(self, img, count):
        h, w, k = img.shape
        digits = []
        for i in range(1, count + 1):
            digit = img[0:h, int((i-1)*w/count):int(i*w/count)]
            self.log_image(digit)
            
            digits.append(digit)
            
        return digits

        
    def crop_image(self, img, center):
        h, w, k = img.shape
        c_x, c_y, x , y = center
        if h/2 > c_y:
            self.log(f'{int(c_y * 2)}')
            img = img[0:int(c_y * 2), 0:w]
        else:
            self.log(int(h - (h - c_y) * 2))
            img = img[int(h - (h - c_y) * 2):h, 0:w]

        
        self.log_image(img)
        return img

    def get_image(self):
        img = cv2.imread(self._path, cv2.IMREAD_COLOR)
        
        h, w, k = img.shape

        img = cv2.resize(img, (int(w * self._ratio ), int (h * self._ratio)))
        img = cv2.rotate(img, cv2.ROTATE_180)
        return img

    def crop_sides(self, img):
        h, w, k = img.shape
        c_x, c_y, x, y = self.find_sample(img, 'samples/sample-right.jpg')
        img = img[0:h, 0:x]
        self.log_image(img)
        
        
        
        img = self.cut_white_sides(img, (6,6))
        

        return img

    def cut_white_sides(self, img, shape, rigth = False):
        h, w, k = img.shape
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        thres = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        kernel = np.ones(shape,np.uint8)
        thres = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernel)
        self.log_image(thres)
        x_left=0
        while x_left<w :
            if thres[int(h/2),x_left]==0:
                break
            x_left+=1
        img = img[:, x_left:w]
        thres = thres[:, x_left:w]
        if rigth:
            h, w, k = img.shape
            x_right=w - 1
            while x_right>=0 :
                if thres[int(h/2),x_right]==0:
                    break
                x_right-=1
            img = img[:, 0: x_right]
       
        self.log_image(img)
        return img

    


    def align_and_crop_with_lines(self, img, lines, center):
        def y_for_line(x, r, theta):
            if theta == 0:
                return np.nan
            return (r - (x * np.cos(theta))) / np.sin(theta)

        h, w, k = img.shape
        x_center, y_center, x, y = center
        rho_above, theta_above = lines[0][0]
        rho_below, theta_below = lines[1][0]
       

        # pts1 = np.float32([
        #     [x_center - 100, y_for_line(x_center - 100, rho_above, theta_above)],
        #     [x_center + 100, y_for_line(x_center + 100, rho_above, theta_above)],
        #     [x_center - 100, y_for_line(x_center - 100, rho_below, theta_below)],
        #     [x_center + 100, y_for_line(x_center + 100, rho_below, theta_below)],
        # ])
        y_above_center = int(y_for_line(x_center, rho_above, theta_above))
        y_below_center = int(y_for_line(x_center, rho_below, theta_below))
        # pts2 = np.float32([
        #     [x_center - 100, y_above_center],
        #     [x_center + 100, y_above_center],
        #     [x_center - 100, y_below_center],
        #     [x_center + 100, y_below_center],
        # ])

        #img = self.draw_lines(img, lines)

        M = cv2.getRotationMatrix2D(
           (0, (rho_below-rho_above)/2+rho_above), theta_above/np.pi*180-90, 1)
        
        img = cv2.warpAffine(img, M, (w, h))

        # M = cv2.getPerspectiveTransform(pts1, pts2)
        # img = cv2.warpPerspective(img, M, (w,h))
        
        self.log_image(img)

        img = img[int(rho_above):int(rho_below), 0:w]
        
        self.log_image(img)

        return img

    def get_lines(self, img, center, thr = 110, thr1 = 100):
        h, w, k = img.shape
        x_center, y_center, x , y = center
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3,3), 1)

        self.log_image(gray, 'gray')


        # for i in range(10, 300, 10):
        #     for j in range(10, i, 10):
        #         edges = cv2.Canny(gray, j, i)
        #         self.log_image(edges, f'test_{j}_{i}')

        edges = cv2.Canny(gray, thr, thr * 2, L2gradient=True)
        self.log_image(edges, 'edges')

        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=thr1)
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

    def find_sample(self, img, path):
        sample = cv2.imread(path)
        sample_h, sample_w, sample_k = sample.shape
        sample = cv2.resize(sample, (int(sample_w * self._ratio ), int (sample_h * self._ratio)))
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

        return x_center, y_center, max_loc[0], max_loc[1]

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
