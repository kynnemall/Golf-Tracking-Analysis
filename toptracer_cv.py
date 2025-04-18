import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# TODO current bug is due to an index matching issue since correct columns
# and numbers are being detected. Fix and test on folder: 250413_driver


class Toptracer:
    """
    Processes screenshot images (in bytes) of Toptracer data using
    template matching to extract metrics into a Pandas DataFrame
    (which is stored under the "df" attribute)
    """

    def __init__(self, img_bytes_list):
        self.unique_indexes = []
        self.raw_indexes = []
        self.rows = []
        self.image_files = img_bytes_list

        self.load_templates()
        self.extract_row_indexes()  # gets indexes
        self.analyse_all_screenshots()
        self.format_data()

    def _gray_img_from_bytes(self, bytes_image):
        gray = Image.open(bytes_image).convert('L')
        gray = np.array(gray)
        return gray

    def read_and_threshold(self, image_path, threshold=200):
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return binary

    def load_templates(self):
        numbers = glob.glob('templates/toptracer/number*')
        num_templates = {
            n[-5]: self.read_and_threshold(n) for n in numbers
        }
        sides = glob.glob('templates/toptracer/side*')
        num_templates.update({
            s[-5]: self.read_and_threshold(s) for s in sides
        })

        metrics = glob.glob('templates/toptracer/metric*')
        metric_keys = [m.split('metric_')[-1][:-4] for m in metrics]
        metrics = {
            key: self.read_and_threshold(metric) for key, metric in zip(
                metric_keys, metrics
            )
        }

        self.number_templates = num_templates
        self.metrics = metrics

    def detect_metrics(self, img):
        max_ = 0
        matches = {
            'hang_time': ['Hang Time', 'Curve', 'Offline'],
            'launch_angle': ['Launch Angle', 'Height', 'Landing Angle'],
            'flat_carry': ['Carry', 'Distance', 'Ball Speed']
        }
        metrics = []
        for metric, template in self.metrics.items():
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            if (res > 0.9).sum() > max_:
                max_ = (res > 0.9).sum()
                metrics = matches[metric]

        return metrics

    def format_data(self):
        dfs = [
            pd.DataFrame(dict_, index=[i]) for i, dict_ in self.data.items()
        ]
        df = pd.concat(dfs).sort_index()  # .dropna(how='all')
        df['Hang Time'] /= 10

        # fix curve and offline greater than 100 -> error matching "R"
        for col in ('Curve', 'Offline'):
            newnums = []
            for value in df[col].values:
                if value <= 0:
                    newnums.append(value)
                elif len(str(value)) > 1:
                    try:
                        new_val = int(str(value)[1:])
                    except ValueError as e:
                        print(f'{e}\nValue:\t{value}')
                        new_val = np.nan
                    newnums.append(new_val)
                else:
                    print(f'Else print: {value}')
            df[col] = newnums

        self.df = df.drop_duplicates()

    def preprocess_image(self, image):
        """Threshold and dilate the image to highlight black text."""
        _, binary = cv2.threshold(
            image, 50, 255, cv2.THRESH_BINARY_INV)  # Detect black text
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=3)
        return dilated

    def is_new_index(self, cropped, threshold=0.95):
        """Use template matching to check if the cropped number is new."""
        for _, _, template in self.unique_indexes:
            result = cv2.matchTemplate(template, cropped, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val > threshold:  # If a similar template exists, return False
                return False
        return True

    def extract_unique_indexes(self, image):
        """Find and store unique index numbers."""
        processed = self.preprocess_image(image)
        contours, _ = cv2.findContours(
            processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x < 200:  # Only consider indexes on the left side
                cropped = image[y:y+h, x:x+w]  # Extract bounding box
                # Normalize size for comparison
                resized = cv2.resize(cropped, (50, 50))

                if self.is_new_index(resized):
                    self.unique_indexes.append((y, x, resized))
                    self.raw_indexes.append(cropped)

    def extract_row_indexes(self):
        """Loop through a list of image paths and extract unique indexes."""
        for bytes_image in self.image_files:
            gray = self._gray_img_from_bytes(bytes_image)[325:1510]
            self.extract_unique_indexes(gray)

        # populate self.data with indexes
        n_indexes = len(self.unique_indexes)
        self.data = {n: {} for n in np.arange(n_indexes)}

    def unified_nms(self, detections, scores=None, threshold=10, axis='x'):
        """
        Generalized Non-Max Suppression.

        Args:
            detections: List of (tag, (x, y)) tuples.
            scores: Optional list of confidence scores.
            threshold: Proximity in pixels to suppress duplicates.
            axis: 'x' or 'y' depending on suppression direction.

        Returns:
            Filtered list of (tag, (x, y)) tuples.
        """
        if not detections:
            return []

        key_index = 0 if axis == 'x' else 1

        if scores is not None:
            zipped = list(zip(detections, scores))
            zipped.sort(key=lambda item: (item[0][1][key_index], -item[1]))
        else:
            zipped = [(det, 1.0) for det in detections]
            zipped.sort(key=lambda item: item[0][1][key_index])

        filtered = []
        for (tag, (x, y)), score in zipped:
            pos = x if axis == 'x' else y
            if all(abs(pos - (fx if axis == 'x' else fy)) > threshold for _, (fx, fy) in filtered):
                filtered.append((tag, (x, y)))

        return filtered

    def find_best_matching_index(self, row_idx_img):
        best_idx = None
        best_score = 0
        scores = []
        preprocessed = self.preprocess_image(row_idx_img)
        contours, _ = cv2.findContours(
            preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # process contour areas, mostly just one but there can be a second
        # contour due to the black bar at the bottom of some screenshots
        contour_areas = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            contour_areas.append(w * h)
        num_contour = contours[np.argmin(contour_areas)]
        x, y, w, h = cv2.boundingRect(num_contour)

        cropped = row_idx_img[y:y+h, x:x+w]  # Extract bounding box
        # Normalize size for comparison
        resized = cv2.resize(cropped, (50, 50))

        for idx, (_, _, template) in enumerate(self.unique_indexes):
            result = cv2.matchTemplate(
                resized, template, cv2.TM_CCOEFF_NORMED
            )
            score = result.max()
            scores.append(score)
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def find_all_matching_indexes(self, image, threshold=0.8):
        """
        Finds all row indexes in the image that match stored unique indexes
        using non-max suppression.
        """
        # preprocess image for template matching
        _, binary = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=3)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_detections = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x < 200:
                cropped = image[y:y+h, x:x+w]
                cropped = cv2.resize(cropped, (50, 50))
                for idx, (iy, ix, template) in enumerate(self.unique_indexes):
                    result = cv2.matchTemplate(
                        template, cropped, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    if max_val > threshold:
                        raw_detections.append((idx, (x, y)))

        filtered = self.unified_nms(raw_detections, axis='y', threshold=15)
        return [(y, idx) for idx, (x, y) in filtered]

    def match_numbers_in_row(self, row_image, base_thresh=0.8):
        digit_thresholds = {}
        detected_numbers = []
        detected_scores = []

        for num, template in self.number_templates.items():
            threshold = digit_thresholds.get(num, base_thresh)
            result = cv2.matchTemplate(
                row_image, template, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= threshold)
            detections = [(num, pt) for pt in zip(*locations[::-1])]
            nms_detections = self.unified_nms(
                detections, axis='x', threshold=10)
            detected_numbers.extend(nms_detections)
            scores = [result[b, a] for _, (a, b) in nms_detections]
            detected_scores.extend(scores)

        final = self.unified_nms(
            detected_numbers, scores=detected_scores, threshold=5, axis='x')
        return final, detected_scores

    def process_screenshot(self, image_bytes, thresh=0.8, row_height=50, plot=False):
        image = self._gray_img_from_bytes(image_bytes)
        matches = self.find_all_matching_indexes(
            image, threshold=thresh
        )
        columns = self.detect_metrics(image)

        if not matches:
            print("No matching index rows found.")
            return

        for row_y, index_id in matches:
            h1, h2 = row_y - 2, row_y + row_height
            row_image = image[h1:h2, 200:]
            row_idx = image[h1:h2, :200]
            index_match = self.find_best_matching_index(row_idx)
            self.rows.append((row_idx, index_match, columns[0]))
            detected_nums, scores = self.match_numbers_in_row(row_image)
            nums = self.group_and_concatenate_tags(detected_nums)

            if plot:
                plt.figure(figsize=(10, 2))
                plt.imshow(row_image, cmap='gray')
                for num, (x, y) in detected_nums:
                    plt.gca().add_patch(
                        plt.Rectangle((x, y), 20, 30, edgecolor='red',
                                      linewidth=2, facecolor='none')
                    )
                    plt.text(x, y - 5, str(num), color='red',
                             fontsize=12, fontweight='bold')
                plt.axis("off")
                plt.show()

            # add results to data dictionary
            assert len(columns) == len(nums), f'{columns}, {nums}'
            for column, num in zip(columns, nums):
                self.data[index_match][column] = num

    def group_and_concatenate_tags(self, detected_numbers, max_distance=100):
        """
        Groups tags by horizontal proximity and concatenates them into strings.

        Args:
            detected_numbers: List of (tag, (x, y)) tuples, sorted by x.
            max_distance: Max horizontal distance in pixels to group tags.

        Returns:
            List of concatenated tag strings.
        """
        if not detected_numbers:
            return []

        groups = []
        current_group = [detected_numbers[0][0]]  # Start with first tag
        last_x = detected_numbers[0][1][0]

        for tag, (x, y) in detected_numbers[1:]:
            if x - last_x <= max_distance:
                current_group.append(tag)
            else:
                groups.append("".join(current_group))
                current_group = [tag]
            last_x = x

        groups.append("".join(current_group))  # Add the last group

        # format numbers with left or right
        nums = []
        for group in groups:
            if group[0] == 'L':
                group = int(group[1:]) * -1 if len(group) > 1 else 0
            elif group[0] == 'R':
                group = int(group[1:]) if len(group) > 1 else 0
            else:
                group = int(group)
            nums.append(group)
        return nums

    def analyse_all_screenshots(self):
        for image_bytes in self.image_files:
            self.process_screenshot(image_bytes)
