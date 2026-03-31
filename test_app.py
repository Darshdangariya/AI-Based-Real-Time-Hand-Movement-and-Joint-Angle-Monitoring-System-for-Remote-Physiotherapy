import unittest
import numpy as np
import cv2
from app import PostureDetector

class TestPostureDetector(unittest.TestCase):
    def setUp(self):
        self.detector = PostureDetector()
    
    def test_calculate_angle(self):
        # Test angle calculation with known points
        a = [0, 0]
        b = [1, 0] 
        c = [1, 1]
        angle = self.detector.calculate_angle(a, b, c)
        self.assertAlmostEqual(angle, 90.0, places=1)
    
    def test_calculate_distance(self):
        # Test distance calculation
        point1 = [0, 0]
        point2 = [3, 4]
        distance = self.detector.calculate_distance(point1, point2)
        self.assertAlmostEqual(distance, 5.0, places=1)
    
    def test_exercise_initialization(self):
        # Test initial state
        self.assertEqual(self.detector.current_exercise, "Neutral")
        self.assertEqual(self.detector.feedback, "")
        self.assertFalse(self.detector.correct_posture)

if __name__ == '__main__':
    unittest.main()
