"""Pure image-region checks for automatic handshake triggering."""


class TargetRegion:
    """Normalized rectangular target with a minimum detected-hand size."""

    def __init__(self, x_min, y_min, x_max, y_max, min_area_ratio):
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.x_max = float(x_max)
        self.y_max = float(y_max)
        self.min_area_ratio = float(min_area_ratio)
        if not (0.0 <= self.x_min < self.x_max <= 1.0):
            raise ValueError(
                "target x bounds must satisfy 0 <= min < max <= 1"
            )
        if not (0.0 <= self.y_min < self.y_max <= 1.0):
            raise ValueError(
                "target y bounds must satisfy 0 <= min < max <= 1"
            )
        if not (0.0 <= self.min_area_ratio <= 1.0):
            raise ValueError("min_area_ratio must be between 0 and 1")

    def pixel_bounds(self, width, height):
        """Return target bounds as integer image coordinates."""
        return (
            int(self.x_min * width), int(self.y_min * height),
            int(self.x_max * width), int(self.y_max * height),
        )

    def accepts(self, hand, width, height):
        """Accept a hand centered in the region and large enough in-frame."""
        if hand is None or hand.bbox is None or width < 1 or height < 1:
            return False
        left, top, right, bottom = self.pixel_bounds(width, height)
        if not (
            left <= hand.center_x <= right
            and top <= hand.center_y <= bottom
        ):
            return False
        x1, y1, x2, y2 = hand.bbox
        area_ratio = max(0, x2 - x1) * max(0, y2 - y1) / float(width * height)
        return area_ratio >= self.min_area_ratio
