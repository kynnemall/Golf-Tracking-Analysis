import cv2
import glob
import numpy as np
import pandas as pd
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
        if string[-1] == 'R':
            return int(string[:-1]) / 10
        else:
            return int(string[:-1]) / -10
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
