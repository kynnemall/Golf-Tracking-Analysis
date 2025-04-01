import cv2
import glob
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None


def load_templates():
    numbers = glob.glob('templates/trackman/number*')
    num_templates = {
        n[-5]: cv2.imread(n, cv2.IMREAD_GRAYSCALE) for n in numbers
    }
    sides = glob.glob('templates/side*')
    num_templates.update({
        s[-5]: cv2.imread(s, cv2.IMREAD_GRAYSCALE) for s in sides
    })
    return num_templates


NUMS = load_templates()


def convert_left_right(string):
    try:
        if string[-1] == 'R':
            return int(string[:-1]) / 10
        else:
            return int(string[:-1]) / -10
    except ValueError:
        print(f"Can't convert to integer: {string}")
    except TypeError:
        print(f"string is float: {string}")


def detect_metrics(img):
    carry = cv2.imread(
        'templates/trackman/metric_carry.jpg', cv2.IMREAD_GRAYSCALE
    )
    res = cv2.matchTemplate(img, carry, cv2.TM_CCOEFF_NORMED)

    if (res > 0.9).sum() > 0:
        metrics = ['Carry', 'Total', 'Ball Speed', 'Height']
    else:
        metrics = ['Launch Angle', 'Launch Dir.', 'Side', 'From Pin']
    return metrics


def detect_numbers(img, metric):
    points = []
    # copy = img.copy()

    if metric in ('Launch Dir.', 'Side'):
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
            # cv2.rectangle(copy, pt, (pt[0] + w, pt[1] + h), 128, 2)
            points.append([label, pt[0], pt[1], res[pt[1], pt[0]]])

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


class Trackman:
    def __init__(self, path, club):
        self.files = sorted(glob.glob(path))
        self.left = []
        self.right = []
        self.data = {}
        for f in self.files:
            df = self.process_screenshot(f, club)
            if df.columns[0] == 'Carry':
                self.left.append(df)
            else:
                self.right.append(df)

        self.combine_left_and_right()

    def process_screenshot(self, filepath, club):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        metrics = detect_metrics(img)
        columns = []
        self.data[filepath] = {'column': [], 'chunks': [], 'strs': []}
        for n, metric in zip((0, 175, 350, 525), metrics):
            # 380:1491 slice to remove column names and footer
            column = img[300:1491, n:n+175]
            chunks, strs = detect_numbers(column, metric)
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
            if metric in ('Launch Dir.', 'Side'):
                df[metric] = df[metric].apply(convert_left_right)
            elif metric in ('Launch Angle', 'From Pin'):
                df[metric] = df[metric]
            else:
                df[metric] = df[metric]

        # convert string to numeric data types
        for col in df.columns[:4]:
            df[col] = df[col].astype(float)

        # correct missing values in last in metrics from launch angle onward
        if df.columns[0] == 'Launch Angle':
            series = [df[col].dropna().reset_index(drop=True)
                      for col in metrics]
            df = pd.concat(series, axis=1)
            df['Club'] = club

            # correct launch angle and pin
            df['Launch Angle'] /= 10
            df['From Pin'] /= 10

        return df

    def combine_left_and_right(self):
        left = pd.concat(self.left)
        right = pd.concat(self.right)

        left.reset_index(drop=True, inplace=True)
        right.reset_index(drop=True, inplace=True)

        combo = pd.concat([left.iloc[:, :4], right], axis=1)
        self.data = combo.drop_duplicates(
            subset=['Carry', 'Total', 'Height'], inplace=True
        )
