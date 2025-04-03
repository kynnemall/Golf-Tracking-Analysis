import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None


def load_templates():
    numbers = glob.glob('templates/toptracer/number*')
    num_templates = {
        n[-5]: cv2.imread(n, cv2.IMREAD_GRAYSCALE) for n in numbers
    }
    sides = glob.glob('templates/toptracer/side*')
    num_templates.update({
        s[-5]: cv2.imread(s, cv2.IMREAD_GRAYSCALE) for s in sides
    })
    metrics = glob.glob('templates/toptracer/metric*')
    metric_keys = [m.split('metric_')[-1][:-4] for m in metrics]
    metrics = {
        key: cv2.imread(metric, cv2.IMREAD_GRAYSCALE) for key, metric in zip(
            metric_keys, metrics
        )
    }
    return num_templates, metrics


NUMS, METRICS = load_templates()


def convert_left_right(string):
    try:
        if string[0] == 'R':
            return int(string[1:])
        else:
            return int(string[1:])
    except ValueError:
        print(f"Can't convert to integer: {string}")
    except TypeError:
        print(f"string is float: {string}")


def detect_metrics(img):
    max_ = 0
    matches = {
        'hang_time': ['Hang Time', 'Curve', 'Offline'],
        'launch_angle': ['Launch Angle', 'Height', 'Landing Angle'],
        'flat_carry': ['Carry', 'Distance', 'Ball Speed']
    }
    metrics = []
    for metric, template in METRICS.items():
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        if (res > 0.9).sum() > max_:
            max_ = (res > 0.9).sum()
            metrics = matches[metric]

    return metrics


def detect_numbers(img, metric):
    points = []
    copy = img.copy()

    if metric in ('Curve', 'Offline'):
        keys = NUMS.keys()
    else:
        keys = [str(n) for n in np.arange(10)]
    for label in keys:
        template = NUMS[label]
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        w, h = template.shape[::-1]

        loc = np.where(res >= 0.8)
        for pt in zip(*loc[::-1]):
            # draw bounding boxes on detected points for debugging
            cv2.rectangle(copy, pt, (pt[0] + w, pt[1] + h), 128, 2)
            points.append([label, pt[0], pt[1], res[pt[1], pt[0]]])

    # plt.imshow(copy)
    points = pd.DataFrame(points, columns=['Label', 'X', 'Y', 'Match'])
    points.sort_values('Y', inplace=True)
    points.reset_index(inplace=True, drop=True)

    # split into chunks based on row pixel location
    chunks = []
    min_val = points['Y'][0]
    chunked = False
    while not chunked:
        chunk = points[
            (points['Y'] - 10 < min_val) & (min_val < points['Y'] + 10)
        ]
        if chunk.size > 0:
            chunk.sort_values('X', inplace=True)
            chunks.append(
                chunk[['Label', 'X', 'Match']].reset_index(drop=True)
            )

        if chunk.index.max() == points.index.max():
            chunked = True
        else:
            min_val = points['Y'][chunk.index.max() + 1]

    # process blocks/chunks of numbers
    strings = []
    for block in chunks:
        string = ''
        min_val = block['X'].min()
        complete = False
        while not complete:
            sub = block[block['X'] <= min_val + 5]
            if sub.size > 0:
                sub = sub[sub['Match'] == sub['Match'].max()]
                letter = sub['Label'].unique()[0]
                string += letter
                block = block[block['X'] > min_val + 5]
                min_val = block['X'].min()
            else:
                complete = True
        strings.append(string)

    return chunks, strings


def detect_rows(img):
    # threshold image to find index keys
    _, binary = cv2.threshold(img, 0, 20, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=3)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through contours to get x and y values for each index
    idx_keys = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if x < 200:
            idx_keys.append([y, x])

    # TODO: continue from here


class Toptracer:
    def __init__(self, path, club):
        self.files = sorted(glob.glob(path))
        self.left = []
        self.middle = []
        self.right = []
        self.indexes = []
        self.data = {}
        self.get_indexes()
        for f in self.files:
            df = self.process_screenshot(f, club)
            if df.columns[0] == 'Carry':
                self.left.append(df)
            elif df.columns[0] == 'Launch Angle':
                self.middle.append(df)
            else:
                self.right.append(df)

        self.combine_sections()

    # TODO
    # update method to find index points and store these as keys in a class attribute
    # use these keys in other images to match rows by index

    def _is_new_index(self, cropped, threshold=0.95):
        """Use template matching to check if the cropped number is new."""
        for _, _, template in self.indexes:
            result = cv2.matchTemplate(template, cropped, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > threshold:  # If a similar template exists, return False
                return False
        return True

    def get_indexes(self):
        for f in self.files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            _, binary = cv2.threshold(
                img, 50, 255, cv2.THRESH_BINARY_INV)  # Detect black text
            kernel = np.ones((3, 3), np.uint8)
            processed = cv2.dilate(binary, kernel, iterations=3)
            contours, _ = cv2.findContours(
                processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if x < 200:  # Only consider indexes on the left side
                    cropped = img[y:y+h, x:x+w]  # Extract bounding box
                    # Normalize size for comparison
                    cropped = cv2.resize(cropped, (50, 50))

                    if self._is_new_index(cropped):
                        self.indexes.append((y, x, cropped))

    def process_screenshot(self, filepath, club):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        metrics = detect_metrics(img)
        img = img[325:1500]
        columns = []
        self.data[filepath] = {'column': [], 'chunks': [], 'strs': []}
        for l, r, metric in zip((0, 150, 350), (150, 350, 550), metrics):
            # 380:1491 slice to remove column names and footer
            column = img[:, l:r]
            chunks, strs = detect_numbers(column[125:, :], metric)
            series = pd.Series(strs, name=metric)
            columns.append(series)

            self.data[filepath]['column'].append(column)
            self.data[filepath]['chunks'].append(chunks)
            self.data[filepath]['strs'].append(strs)

        # prepare the data
        df = pd.concat(columns, axis=1)
        df.columns = metrics
        df['Club'] = club

        for metric in metrics:
            if metric in ('Curve', 'Offline'):
                df[metric] = df[metric].apply(convert_left_right)
            elif metric in ('Launch Angle', 'From Pin'):
                df[metric] = df[metric]
            else:
                df[metric] = df[metric]

        # convert string to numeric data types
        for col in df.columns[:-1]:
            df[col] = df[col].astype(float)

        # correct missing values in last in metrics from launch angle onward
        if df.columns[0] == 'Hang Time':
            series = [df[col].dropna().reset_index(drop=True)
                      for col in metrics]
            df = pd.concat(series, axis=1)
            df['Club'] = club
            df['Hang Time'] /= 10  # correct hang time to decimal

        return df

    def combine_sections(self):
        left = pd.concat(self.left)
        right = pd.concat(self.right)
        middle = pd.concat(self.middle)

        left.reset_index(drop=True, inplace=True)
        right.reset_index(drop=True, inplace=True)
        middle.reset_index(drop=True, inplace=True)

        combo = pd.concat(
            [left.iloc[:, :3], middle.iloc[:, :3], right], axis=1)
        self.df = combo.drop_duplicates(subset=combo.columns[:9])
