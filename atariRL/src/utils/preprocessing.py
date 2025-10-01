"""Frame preprocessing utilities for Atari environments."""

import cv2
import numpy as np
from collections import deque
from typing import Tuple


class AtariPreprocessor:
    """Preprocessor for Atari game frames.
    
    This class handles frame preprocessing including grayscale conversion,
    resizing, and frame stacking for temporal information.
    """
    
    def __init__(self, frame_size: Tuple[int, int] = (84, 84), frame_stack: int = 4):
        """Initialize the preprocessor.
        
        Args:
            frame_size: Target size for resized frames (width, height)
            frame_stack: Number of frames to stack together
        """
        self.frame_size = frame_size
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess a single frame.
        
        Args:
            frame: Raw RGB frame from environment
            
        Returns:
            Preprocessed grayscale frame normalized to [0, 1]
        """
        # Convert RGB to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize to target size
        resized_frame = cv2.resize(gray_frame, self.frame_size)
        
        # Normalize pixel values to [0, 1]
        normalized_frame = resized_frame.astype(np.float32) / 255.0
        
        return normalized_frame
    
    def reset(self, initial_frame: np.ndarray) -> np.ndarray:
        """Reset the preprocessor with an initial frame.
        
        Args:
            initial_frame: First frame of a new episode
            
        Returns:
            Stacked frames array
        """
        processed_frame = self.preprocess_frame(initial_frame)
        
        # Fill the frame stack with the initial frame
        for _ in range(self.frame_stack):
            self.frames.append(processed_frame)
        
        return np.array(self.frames)
    
    def step(self, frame: np.ndarray) -> np.ndarray:
        """Process a new frame and update the frame stack.
        
        Args:
            frame: New frame from environment step
            
        Returns:
            Updated stacked frames array
        """
        processed_frame = self.preprocess_frame(frame)
        self.frames.append(processed_frame)
        
        return np.array(self.frames)
    
    def get_frame_shape(self) -> Tuple[int, int, int]:
        """Get the shape of stacked frames.
        
        Returns:
            Shape tuple (frame_stack, height, width)
        """
        return (self.frame_stack, self.frame_size[1], self.frame_size[0])