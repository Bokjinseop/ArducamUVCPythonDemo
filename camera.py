from datetime import datetime
import time
import cv2
import numpy as np

class Camera:

    def __init__(self, index=0, selector=cv2.CAP_ANY) -> None:
        self.index = index
        self.selector = selector
        self.cap = None
        self.width = None
        self.height = None
        self.fps = None

    def open(self):
        self.cap = cv2.VideoCapture(self.index, self.selector)

        if self.width and self.height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if self.fps:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
    
    def set_width(self, width):
        self.width = width
    
    def set_height(self, height):
        self.height = height
    
    def set_fps(self, fps):
        self.fps = fps

    def set_focus(self, val):
        self.cap.set(cv2.CAP_PROP_FOCUS, val)
        
    def read(self):
        return self.cap.read()

    def reStart(self):
        self.release()
        time.sleep(0.5)
        self.open()

    def release(self):
        self.cap.release()

    def isOpened(self):
        return self.cap.isOpened()
        
    
    @staticmethod
    def _focus_measure(gray_frame):
        """Compute sharpness score of a grayscale image using variance of Laplacian."""
        lap = cv2.Laplacian(gray_frame, cv2.CV_64F)
        return lap.var()
    
    @staticmethod
    def _tenengrad(gray):
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return np.mean(gx**2 + gy**2)

    def autofocus_brent(
        self,
        low=0,
        high=1023,
        tol=5.0,
        max_iter=20,
        delay=0.1,
        scale=0.5,
        crop_ratio=0.6,
        show=False
    ):
        """
        Expert autofocus via Brent's method (combines parabolic interpolation and golden section):
        - Optimize a unimodal focus measure (_tenengrad) over [low, high].
        - tol: convergence tolerance in focus units.
        - max_iter: maximum iterations.
        - crop_ratio: fraction of center region to analyze.
        - scale: downscale factor for speed.
        - show: preview ROI during search.
        Returns (best_focus, best_score).
        """
        if self.cap is None or not self.cap.isOpened():
            self.open()

        gr = (np.sqrt(5) - 1) / 2 # 0.618033
        print(gr)
        a, b = float(low), float(high)
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        print(f"a={a:.1f}, b={b:.1f}, c={c:.1f}, d={d:.1f}")

        def measure_focus(pos):
            # set and wait
            self.set_focus(pos)
            time.sleep(delay)
            ret, frame = self.read()
            
            if not ret:
                return -np.inf
            # crop center
            h, w = frame.shape[:2]
            ch, cw = int(h * crop_ratio), int(w * crop_ratio)
            y0, x0 = (h - ch)//2, (w - cw)//2
            roi = frame[y0:y0+ch, x0:x0+cw]
            if show:
                disp = cv2.resize(roi, (cw//2, ch//2))
                cv2.imshow('AF Preview', disp)
                cv2.waitKey(1)
            if scale != 1.0:
                roi = cv2.resize(roi, (int(cw*scale), int(ch*scale)), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # time_str = datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f')[:-3]
            # output_path = f"./images/test_{time_str}_{pos}.jpg"
            # cv2.imwrite(f"{output_path}", frame)
            return Camera._tenengrad(gray)

        fc = measure_focus(c)
        fd = measure_focus(d)
        best_focus, best_score = (c, fc) if fc > fd else (d, fd)
        print(f"Iter {0}: a={a:.1f}, b={b:.1f}, c={c:.1f}, d={d:.1f}, fc={fc:.1f}, fd={fd:.1f}, best={best_focus:.1f} ({best_score:.1f})")

        for i in range(max_iter):
            if abs(b - a) < tol:
                break
            if fc > fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = measure_focus(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = measure_focus(d)

            # update best
            if fc > best_score:
                best_focus, best_score = c, fc
            if fd > best_score:
                best_focus, best_score = d, fd
                
            print(f"Iter {i+1}: a={a:.1f}, b={b:.1f}, c={c:.1f}, d={d:.1f}, fc={fc:.1f}, fd={fd:.1f}, best={best_focus:.1f} ({best_score:.1f})")

            if (best_score > 18000):
                break

        # final set
        self.set_focus(best_focus)
        if show:
            cv2.destroyWindow('AF Preview')
        return int(best_focus), best_score
    
    def diagnostic_sweep(cam, low=0, high=1023, steps=50, delay=0.1):
        import matplotlib.pyplot as plt

        vals = np.linspace(low, high, steps, dtype=int)
        scores = []
        for v in vals:
            cam.set_focus(v)
            time.sleep(delay)
            ret, frame = cam.read()
            if not ret:
                scores.append(0)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            scores.append(Camera._tenengrad(gray))
        # 그래프 그리기
        plt.figure()
        plt.plot(vals, scores, marker='o')
        plt.title('Focus vs Sharpness')
        plt.xlabel('Focus Value')
        plt.ylabel('Sharpness Score')
        plt.show()

    '''
    @staticmethod
    def _focus_measure(gray_frame):
        """Compute sharpness score of a grayscale image using variance of Laplacian."""
        lap = cv2.Laplacian(gray_frame, cv2.CV_64F)
        return lap.var()

    def autofocus_divide_conquer(
        self,
        low=0,
        high=1023,
        iterations=5,
        delay=0.1,
        scale=0.5,
        crop_ratio=0.5,
        show=True
    ):
        """
        Autofocus using divide-and-conquer with corrected range narrowing:
        - Sample focus at five points: low, 1/4, 2/4, 3/4, high.
        - Downscale full frame before grayscale conversion for speed.
        - Optionally display each raw frame if show=True.
        - Narrow search range based on best index:
          * idx=0: [chosen, chosen+L/4]
          * idx=4: [chosen-L/4, chosen]
          * else: [chosen-L/8, chosen+L/8]
        - Repeat for given iterations.
        Returns best focus value and its score.
        """
        if not self.isOpened():
            self.open()

        best_focus = None
        best_score = -1

        #if show:
        #    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
        low_bound, high_bound = low, high
        for i in range(iterations):
            L = high - low
            quarter = L / 4
            points = [low,
                      low + quarter,
                      low + 2*quarter,
                      low + 3*quarter,
                      high]
            candidates = [int(p) for p in points]
            scores = []
            
            best_focus = None
            best_score = -1

            for v in candidates:
                self.set_focus(v)
                time.sleep(delay)
                ret, frame = self.read()
                if not ret:
                    scores.append(-1)
                    continue

                if show:
                    cv2.imshow('video', frame)
                    cv2.waitKey(1)

                # Crop central region
                h, w = frame.shape[:2]
                ch, cw = int(h*crop_ratio), int(w*crop_ratio)
                y1, x1 = (h-ch)//2, (w-cw)//2
                crop = frame[y1:y1+ch, x1:x1+cw]
                    
                # Downscale crop
                if scale != 1.0:
                    ch2, cw2 = crop.shape[:2]
                    proc = cv2.resize(crop, (int(cw2*scale), int(ch2*scale)), interpolation=cv2.INTER_AREA)
                else:
                    proc = crop

                gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
                score = self._focus_measure(gray)
                scores.append(score)
                print(f"Iter {i+1}: Focus={v}, Score={score:.2f}")
                if score > best_score:
                    best_score, best_focus = score, v

            idx = int(np.argmax(scores))
            chosen = candidates[idx]
            print(f"Chosen point: {chosen} (idx={idx})")

            # Corrected range narrowing using chosen
            if idx == 0 and chosen == low:
                new_low, new_high = chosen, chosen + quarter
            elif idx == 4 and chosen == high:
                new_low, new_high = high - quarter, high
            else:
                half_q = quarter / 2
                new_low, new_high = chosen - half_q, chosen + half_q

            low = max(low_bound, new_low)
            high = min(high_bound, new_high)
            print(f"New range: [{int(low)}, {int(high)}]")

        #if show:
        #   cv2.destroyWindow('focus_preview')

        print(f"Best focus after D&C: {best_focus} (Score={best_score:.2f})\n")
        self.set_focus(best_focus)
        return best_focus, best_score
        '''