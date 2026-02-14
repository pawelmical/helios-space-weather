"""
HELIOS Validation - CME Detection Module
==========================================
Image-based CME detection using running-difference, thresholding, 
and morphological cleaning.

Supports:
- SOHO/LASCO coronagraph images
- STEREO A/B COR1/COR2 images
- Synthetic test images

Author: HELIOS Team
Date: January 2026
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import warnings

# Try importing image processing libraries
try:
    from scipy import ndimage
    from scipy.ndimage import binary_opening, binary_closing, label
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available - using simplified detection")

try:
    from skimage import morphology, measure  # type: ignore
    from skimage.filters import threshold_otsu  # type: ignore
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    warnings.warn("scikit-image not available - using basic morphology")


@dataclass
class DetectionResult:
    """Container for CME detection results."""
    detected: bool
    first_detection_time: Optional[datetime]
    detection_frames: List[int]
    confidence_scores: List[float]
    per_frame_pixels_above_threshold: List[float]
    parameters: Dict
    
    def to_dict(self) -> Dict:
        return {
            'detected': self.detected,
            'first_detection_time': self.first_detection_time.isoformat() if self.first_detection_time else None,
            'num_detection_frames': len(self.detection_frames),
            'mean_confidence': np.mean(self.confidence_scores) if self.confidence_scores else 0.0,
            'max_confidence': max(self.confidence_scores) if self.confidence_scores else 0.0,
            **self.parameters
        }


class CMEDetector:
    """
    CME detector using running-difference and morphological analysis.
    
    Parameters
    ----------
    method : str
        Detection method: 'running_diff', 'base_diff', 'combined'
    diff_threshold : float
        Threshold for difference image (0-1 scale)
    min_area_px : int
        Minimum connected area in pixels to count as detection
    morph_kernel_size : int
        Size of morphological structuring element
    min_frames_for_detection : int
        Minimum number of consecutive frames with detection
    """
    
    def __init__(
        self,
        method: str = 'running_diff',
        diff_threshold: float = 0.15,
        min_area_px: int = 500,
        morph_kernel_size: int = 5,
        min_frames_for_detection: int = 2,
        use_adaptive_threshold: bool = True
    ):
        self.method = method
        self.diff_threshold = diff_threshold
        self.min_area_px = min_area_px
        self.morph_kernel_size = morph_kernel_size
        self.min_frames_for_detection = min_frames_for_detection
        self.use_adaptive_threshold = use_adaptive_threshold
        
    def detect_cme(
        self,
        image_stack: np.ndarray,
        timestamps: Optional[List[datetime]] = None,
        mask: Optional[np.ndarray] = None
    ) -> DetectionResult:
        """
        Detect CME in a stack of coronagraph images.
        
        Parameters
        ----------
        image_stack : np.ndarray
            Stack of images with shape (n_frames, height, width)
        timestamps : list of datetime, optional
            Timestamps for each frame
        mask : np.ndarray, optional
            Boolean mask for valid pixels (True = valid)
            
        Returns
        -------
        result : DetectionResult
            Detection results including timing and confidence
        """
        n_frames = image_stack.shape[0]
        
        if timestamps is None:
            # Generate placeholder timestamps
            timestamps = [datetime.now() for _ in range(n_frames)]
        
        if mask is None:
            mask = np.ones(image_stack.shape[1:], dtype=bool)
            
        # Compute difference images
        if self.method == 'running_diff':
            diff_images = self._running_difference(image_stack)
        elif self.method == 'base_diff':
            diff_images = self._base_difference(image_stack)
        else:
            diff_images = self._running_difference(image_stack)
        
        # Analyze each difference frame
        detection_frames = []
        confidence_scores = []
        pixels_above_threshold = []
        
        for i, diff_img in enumerate(diff_images):
            # Apply mask
            masked_diff = diff_img * mask
            
            # Normalize to 0-1 range
            if masked_diff.max() > masked_diff.min():
                normalized = (masked_diff - masked_diff.min()) / (masked_diff.max() - masked_diff.min())
            else:
                normalized = np.zeros_like(masked_diff)
            
            # Threshold
            if self.use_adaptive_threshold and SKIMAGE_AVAILABLE:
                try:
                    thresh = threshold_otsu(normalized[mask])
                    thresh = max(thresh, self.diff_threshold)
                except:
                    thresh = self.diff_threshold
            else:
                thresh = self.diff_threshold
                
            binary = normalized > thresh
            
            # Morphological cleaning
            cleaned = self._morphological_clean(binary)
            
            # Find connected regions
            detected, area, confidence = self._analyze_regions(cleaned, mask)
            
            # Record statistics
            valid_pixels = mask.sum()
            frac_above = (cleaned & mask).sum() / valid_pixels if valid_pixels > 0 else 0
            pixels_above_threshold.append(frac_above)
            
            if detected:
                detection_frames.append(i + 1)  # 1-indexed frame number
                confidence_scores.append(confidence)
        
        # Determine if CME was detected (require consecutive frames)
        cme_detected = self._check_consecutive_detections(
            detection_frames, 
            self.min_frames_for_detection
        )
        
        # Get first detection time
        first_time = None
        if cme_detected and detection_frames:
            first_frame_idx = detection_frames[0] - 1
            first_time = timestamps[min(first_frame_idx + 1, len(timestamps) - 1)]
        
        return DetectionResult(
            detected=cme_detected,
            first_detection_time=first_time,
            detection_frames=detection_frames,
            confidence_scores=confidence_scores,
            per_frame_pixels_above_threshold=pixels_above_threshold,
            parameters={
                'method': self.method,
                'diff_threshold': self.diff_threshold,
                'min_area_px': self.min_area_px,
                'n_frames_analyzed': len(diff_images)
            }
        )
    
    def _running_difference(self, image_stack: np.ndarray) -> np.ndarray:
        """Compute running difference images."""
        n_frames = image_stack.shape[0]
        diff_images = []
        
        for i in range(1, n_frames):
            diff = image_stack[i].astype(float) - image_stack[i-1].astype(float)
            # Clip negatives (we're looking for brightness increases)
            diff = np.clip(diff, 0, None)
            diff_images.append(diff)
        
        return np.array(diff_images)
    
    def _base_difference(self, image_stack: np.ndarray) -> np.ndarray:
        """Compute base difference images (subtract first frame)."""
        base = image_stack[0].astype(float)
        diff_images = []
        
        for i in range(1, len(image_stack)):
            diff = image_stack[i].astype(float) - base
            diff = np.clip(diff, 0, None)
            diff_images.append(diff)
        
        return np.array(diff_images)
    
    def _morphological_clean(self, binary: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean binary image."""
        if SCIPY_AVAILABLE:
            struct = np.ones((self.morph_kernel_size, self.morph_kernel_size))
            # Opening removes small objects
            cleaned = binary_opening(binary, structure=struct)
            # Closing fills small holes
            cleaned = binary_closing(cleaned, structure=struct)
        else:
            # Simplified cleaning without scipy
            cleaned = binary.copy()
        
        return cleaned
    
    def _analyze_regions(
        self, 
        binary: np.ndarray, 
        mask: np.ndarray
    ) -> Tuple[bool, int, float]:
        """
        Analyze connected regions in binary image.
        
        Returns
        -------
        detected : bool
            Whether a CME-like feature was detected
        area : int
            Area of largest region in pixels
        confidence : float
            Detection confidence (0-1)
        """
        if SKIMAGE_AVAILABLE:
            labeled = measure.label(binary)
            regions = measure.regionprops(labeled)
            
            if not regions:
                return False, 0, 0.0
            
            # Get largest region
            largest = max(regions, key=lambda r: r.area)
            area = largest.area
            
        elif SCIPY_AVAILABLE:
            labeled, n_features = label(binary)
            if n_features == 0:
                return False, 0, 0.0
            
            # Find largest connected component
            component_sizes = ndimage.sum(binary, labeled, range(1, n_features + 1))
            area = int(max(component_sizes)) if len(component_sizes) > 0 else 0
        else:
            # Fallback: just count pixels
            area = binary.sum()
        
        # Check if area meets minimum threshold
        detected = area >= self.min_area_px
        
        # Calculate confidence based on area and shape
        valid_area = mask.sum()
        confidence = min(1.0, area / (valid_area * 0.1)) if detected else 0.0
        
        return detected, area, confidence
    
    def _check_consecutive_detections(
        self, 
        detection_frames: List[int], 
        min_consecutive: int
    ) -> bool:
        """Check if there are enough consecutive detection frames."""
        if len(detection_frames) < min_consecutive:
            return False
        
        # Check for consecutive frames
        consecutive = 1
        max_consecutive = 1
        
        for i in range(1, len(detection_frames)):
            if detection_frames[i] == detection_frames[i-1] + 1:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 1
        
        return max_consecutive >= min_consecutive


def generate_synthetic_cme_images(
    n_frames: int = 20,
    image_size: Tuple[int, int] = (512, 512),
    cme_start_frame: int = 5,
    cme_speed_px_per_frame: float = 15.0,
    cme_width_deg: float = 60.0,
    noise_level: float = 0.05,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, List[datetime], np.ndarray]:
    """
    Generate synthetic coronagraph images with a CME.
    
    For testing the detection algorithm without real data.
    
    Parameters
    ----------
    n_frames : int
        Number of frames to generate
    image_size : tuple
        Image dimensions (height, width)
    cme_start_frame : int
        Frame number when CME first appears
    cme_speed_px_per_frame : float
        CME expansion speed in pixels per frame
    cme_width_deg : float
        Angular width of CME in degrees
    noise_level : float
        Standard deviation of Gaussian noise
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    images : np.ndarray
        Stack of synthetic images
    timestamps : list
        Timestamps for each frame
    mask : np.ndarray
        Occulter mask (True = valid pixels)
    """
    if seed is not None:
        np.random.seed(seed)
    
    h, w = image_size
    center = (h // 2, w // 2)
    
    # Create coordinate grids
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    theta = np.arctan2(y - center[0], x - center[1])
    
    # Create occulter mask (central disk blocked)
    inner_radius = min(h, w) * 0.15
    outer_radius = min(h, w) * 0.48
    mask = (r > inner_radius) & (r < outer_radius)
    
    # Create background (K-corona model: decreases with r^-2.5)
    background = np.zeros((h, w))
    background[mask] = 1.0 / (r[mask] / inner_radius) ** 2.5
    background = background / background.max() * 0.5  # Normalize
    
    # CME direction (position angle)
    cme_pa = np.deg2rad(270)  # West limb
    cme_half_width = np.deg2rad(cme_width_deg / 2)
    
    images = []
    base_time = datetime(2000, 7, 14, 10, 0, 0)
    timestamps = []
    
    for i in range(n_frames):
        # Base corona with noise
        frame = background.copy()
        frame += np.random.normal(0, noise_level, (h, w))
        
        # Add CME if after start frame
        if i >= cme_start_frame:
            frames_since_cme = i - cme_start_frame + 1
            cme_front_radius = inner_radius + cme_speed_px_per_frame * frames_since_cme
            
            # CME is a bright arc
            in_cme_angle = np.abs(theta - cme_pa) < cme_half_width
            in_cme_radial = (r > inner_radius) & (r < cme_front_radius)
            cme_region = in_cme_angle & in_cme_radial & mask
            
            # CME brightness decreases with radius
            cme_brightness = np.zeros((h, w))
            cme_brightness[cme_region] = 0.3 * (1 - (r[cme_region] - inner_radius) / 
                                                  (cme_front_radius - inner_radius + 1))
            frame += cme_brightness
        
        # Apply mask (set occluded region to 0)
        frame = frame * mask
        
        images.append(frame)
        timestamps.append(base_time + timedelta(minutes=i * 12))  # 12-minute cadence
    
    return np.array(images), timestamps, mask


def detect_cme_simple(
    image_stack: np.ndarray,
    threshold: float = 0.15,
    min_area: int = 500
) -> Dict:
    """
    Simple CME detection function for quick analysis.
    
    Parameters
    ----------
    image_stack : np.ndarray
        Stack of images (n_frames, height, width)
    threshold : float
        Detection threshold
    min_area : int
        Minimum detection area in pixels
        
    Returns
    -------
    result : dict
        Detection results
    """
    detector = CMEDetector(
        diff_threshold=threshold,
        min_area_px=min_area
    )
    result = detector.detect_cme(image_stack)
    return result.to_dict()


def analyze_detection_sequence(
    detections: List[DetectionResult],
    instrument_names: List[str]
) -> Dict:
    """
    Analyze a sequence of detections from multiple instruments.
    
    Parameters
    ----------
    detections : list of DetectionResult
        Detection results from each instrument
    instrument_names : list of str
        Names of instruments
        
    Returns
    -------
    analysis : dict
        Multi-instrument detection analysis
    """
    analysis = {
        'n_instruments': len(detections),
        'n_detections': sum(1 for d in detections if d.detected),
        'instruments_with_detection': [],
        'first_detection_time': None,
        'first_detection_instrument': None,
        'detection_times': {},
    }
    
    earliest_time = None
    earliest_instrument = None
    
    for i, (det, name) in enumerate(zip(detections, instrument_names)):
        if det.detected:
            analysis['instruments_with_detection'].append(name)
            analysis['detection_times'][name] = det.first_detection_time
            
            if det.first_detection_time:
                if earliest_time is None or det.first_detection_time < earliest_time:
                    earliest_time = det.first_detection_time
                    earliest_instrument = name
    
    analysis['first_detection_time'] = earliest_time
    analysis['first_detection_instrument'] = earliest_instrument
    
    return analysis


def create_detection_report(
    event_detections: Dict[str, Dict[str, DetectionResult]],
    ground_truth: Optional[Dict] = None
):
    """
    Create a detection report DataFrame.
    
    Parameters
    ----------
    event_detections : dict
        Nested dict: {event_id: {instrument: DetectionResult}}
    ground_truth : dict, optional
        Ground truth labels: {event_id: bool}
        
    Returns
    -------
    df : pd.DataFrame
        Detection report
    """
    import pandas as pd
    
    rows = []
    
    for event_id, instrument_results in event_detections.items():
        for instrument, result in instrument_results.items():
            row = {
                'event_id': event_id,
                'instrument': instrument,
                'detected': result.detected,
                'detection_time': result.first_detection_time,
                'n_frames_detected': len(result.detection_frames),
                'mean_confidence': np.mean(result.confidence_scores) if result.confidence_scores else 0.0,
                'max_confidence': max(result.confidence_scores) if result.confidence_scores else 0.0,
            }
            
            if ground_truth is not None:
                gt = ground_truth.get(event_id, None)
                if gt is not None:
                    row['ground_truth'] = gt
                    row['correct'] = result.detected == gt
            
            rows.append(row)
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 60)
    print("HELIOS Detection Module - Test")
    print("=" * 60)
    
    # Generate synthetic test data
    print("\nGenerating synthetic CME images...")
    images, timestamps, mask = generate_synthetic_cme_images(
        n_frames=20,
        image_size=(256, 256),
        cme_start_frame=5,
        seed=42
    )
    print(f"  Generated {len(images)} frames of size {images.shape[1:]}") 
    
    # Run detection
    print("\nRunning CME detection...")
    detector = CMEDetector(
        method='running_diff',
        diff_threshold=0.10,
        min_area_px=200,
        min_frames_for_detection=2
    )
    
    result = detector.detect_cme(images, timestamps, mask)
    
    print(f"\n  Detection Result:")
    print(f"    Detected: {result.detected}")
    print(f"    First Detection Frame: {result.detection_frames[0] if result.detection_frames else 'N/A'}")
    print(f"    Total Detection Frames: {len(result.detection_frames)}")
    print(f"    Mean Confidence: {np.mean(result.confidence_scores):.3f}" if result.confidence_scores else "    Mean Confidence: N/A")
    
    print("\n" + "=" * 60)
    print("Test completed!")
