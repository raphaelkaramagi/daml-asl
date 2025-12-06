#!/usr/bin/env python3
"""
ASL Alphabet Recognition - Demo Script
=======================================
Quick demo to test both trained models on sample images.

Usage:
    python demo.py                    # Interactive menu
    python demo.py --test             # Test on all test images
    python demo.py path/to/image.jpg  # Test on a specific image
"""

import os
import sys
import warnings

# Suppress warnings before importing other libraries
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import numpy as np
import cv2

# Suppress TensorFlow logging
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

from tensorflow import keras
import joblib
import mediapipe as mp

# Paths
MODELS_DIR = "models"
DATA_DIR = "data"
TEST_DIR = "data/asl_alphabet_test/asl_alphabet_test"

# Model paths
RESNET_MODEL = os.path.join(MODELS_DIR, "best_asl_resnet50_phase2.h5")
LANDMARK_MODEL = os.path.join(DATA_DIR, "nn_landmark_model.keras")
LABEL_ENCODER = os.path.join(DATA_DIR, "label_encoder.joblib")
SCALER = os.path.join(DATA_DIR, "scaler.joblib")

# Class names
CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
               'del', 'nothing', 'space']

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print styled header."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║           ASL Alphabet Recognition Demo                  ║")
    print("║              Two-Approach Comparison                     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")


def print_menu():
    """Print interactive menu."""
    print(f"\n{Colors.YELLOW}Choose an option:{Colors.END}")
    print(f"  {Colors.BOLD}1{Colors.END} - Run full test set evaluation")
    print(f"  {Colors.BOLD}2{Colors.END} - Test a specific image")
    print(f"  {Colors.BOLD}3{Colors.END} - Show model info")
    print(f"  {Colors.BOLD}q{Colors.END} - Quit")
    print()


def load_models(quiet=False):
    """Load both models and preprocessing tools."""
    if not quiet:
        print(f"\n{Colors.BLUE}Loading models...{Colors.END}")
    
    models = {}
    
    # Load ResNet50
    if os.path.exists(RESNET_MODEL):
        models['resnet'] = keras.models.load_model(RESNET_MODEL, compile=False)
        if not quiet:
            print(f"  {Colors.GREEN}✓{Colors.END} ResNet50 loaded")
    else:
        if not quiet:
            print(f"  {Colors.RED}✗{Colors.END} ResNet50 not found")
    
    # Load Landmark NN
    if os.path.exists(LANDMARK_MODEL) and os.path.exists(LABEL_ENCODER) and os.path.exists(SCALER):
        models['landmark'] = keras.models.load_model(LANDMARK_MODEL, compile=False)
        models['label_encoder'] = joblib.load(LABEL_ENCODER)
        models['scaler'] = joblib.load(SCALER)
        if not quiet:
            print(f"  {Colors.GREEN}✓{Colors.END} Landmark NN loaded")
    else:
        if not quiet:
            print(f"  {Colors.RED}✗{Colors.END} Landmark NN not found")
    
    return models


def extract_landmarks(image_rgb):
    """Extract hand landmarks using MediaPipe."""
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        results = hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            wrist = hand_landmarks.landmark[0]
            
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
            return np.array(landmarks)
    return None


def predict_image(image_path, models):
    """Predict ASL letter from an image using both models."""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictions = {}
    
    # ResNet50 prediction
    if 'resnet' in models:
        img_resized = cv2.resize(image_rgb, (96, 96))
        img_normalized = img_resized / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        pred = models['resnet'].predict(img_batch, verbose=0)
        pred_idx = np.argmax(pred[0])
        confidence = pred[0][pred_idx]
        predictions['resnet'] = (CLASS_NAMES[pred_idx], confidence)
    
    # Landmark NN prediction
    if 'landmark' in models:
        landmarks = extract_landmarks(image_rgb)
        if landmarks is not None:
            landmarks_scaled = models['scaler'].transform([landmarks])
            pred = models['landmark'].predict(landmarks_scaled, verbose=0)
            pred_idx = np.argmax(pred[0])
            confidence = pred[0][pred_idx]
            label = models['label_encoder'].inverse_transform([pred_idx])[0]
            predictions['landmark'] = (label, confidence)
        else:
            predictions['landmark'] = ("NO_HAND", 0.0)
    
    return predictions


def demo_single_image(image_path, models):
    """Demo on a single image."""
    print(f"\n{Colors.BLUE}Processing:{Colors.END} {image_path}")
    
    predictions = predict_image(image_path, models)
    
    if predictions:
        print(f"\n{Colors.BOLD}Results:{Colors.END}")
        print("─" * 40)
        if 'resnet' in predictions:
            label, conf = predictions['resnet']
            conf_bar = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
            print(f"  ResNet50:    {Colors.CYAN}{label:8s}{Colors.END} [{conf_bar}] {conf*100:5.1f}%")
        if 'landmark' in predictions:
            label, conf = predictions['landmark']
            if label == "NO_HAND":
                print(f"  Landmark NN: {Colors.RED}NO HAND DETECTED{Colors.END}")
            else:
                conf_bar = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
                print(f"  Landmark NN: {Colors.CYAN}{label:8s}{Colors.END} [{conf_bar}] {conf*100:5.1f}%")
        print("─" * 40)
    else:
        print(f"{Colors.RED}Could not load image{Colors.END}")


def demo_test_set(models):
    """Demo on all test images with clean output."""
    if not os.path.exists(TEST_DIR):
        print(f"{Colors.RED}Test directory not found: {TEST_DIR}{Colors.END}")
        return
    
    print(f"\n{Colors.BLUE}Testing on all images in test set...{Colors.END}\n")
    
    # Collect results
    results = []
    files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(('.jpg', '.png'))])
    
    for i, filename in enumerate(files):
        image_path = os.path.join(TEST_DIR, filename)
        true_label = filename.replace('_test.jpg', '').replace('_test.png', '')
        predictions = predict_image(image_path, models)
        
        resnet_pred = predictions.get('resnet', (None, 0))
        landmark_pred = predictions.get('landmark', (None, 0))
        
        results.append({
            'true': true_label,
            'resnet': resnet_pred,
            'landmark': landmark_pred
        })
        
        # Progress indicator
        progress = (i + 1) / len(files)
        bar = "█" * int(progress * 30) + "░" * (30 - int(progress * 30))
        print(f"\r  Progress: [{bar}] {i+1}/{len(files)}", end='', flush=True)
    
    print("\n")
    
    # Print results table
    print(f"{'─' * 60}")
    print(f"{Colors.BOLD}{'Letter':<10}{'ResNet50':<20}{'Landmark NN':<20}{Colors.END}")
    print(f"{'─' * 60}")
    
    correct_resnet = 0
    correct_landmark = 0
    
    for r in results:
        true = r['true']
        
        # ResNet result
        if r['resnet'][0]:
            resnet_ok = r['resnet'][0] == true
            if resnet_ok:
                correct_resnet += 1
            resnet_str = f"{Colors.GREEN}✓{Colors.END} {r['resnet'][0]}" if resnet_ok else f"{Colors.RED}✗{Colors.END} {r['resnet'][0]}"
        else:
            resnet_str = "—"
        
        # Landmark result
        if r['landmark'][0] and r['landmark'][0] != "NO_HAND":
            landmark_ok = r['landmark'][0] == true
            if landmark_ok:
                correct_landmark += 1
            landmark_str = f"{Colors.GREEN}✓{Colors.END} {r['landmark'][0]}" if landmark_ok else f"{Colors.RED}✗{Colors.END} {r['landmark'][0]}"
        elif r['landmark'][0] == "NO_HAND":
            landmark_str = f"{Colors.YELLOW}⚠ NO HAND{Colors.END}"
        else:
            landmark_str = "—"
        
        print(f"  {true:<8} {resnet_str:<28} {landmark_str:<28}")
    
    total = len(results)
    print(f"{'─' * 60}")
    
    # Summary
    print(f"\n{Colors.BOLD}Summary:{Colors.END}")
    print(f"{'─' * 40}")
    
    resnet_acc = correct_resnet / total * 100
    landmark_acc = correct_landmark / total * 100
    
    print(f"  ResNet50:    {Colors.CYAN}{correct_resnet}/{total}{Colors.END} ({resnet_acc:.1f}%)")
    print(f"  Landmark NN: {Colors.CYAN}{correct_landmark}/{total}{Colors.END} ({landmark_acc:.1f}%)")
    print(f"{'─' * 40}")
    
    if resnet_acc > landmark_acc:
        print(f"\n  {Colors.GREEN}🏆 ResNet50 wins on test set!{Colors.END}")
    elif landmark_acc > resnet_acc:
        print(f"\n  {Colors.GREEN}🏆 Landmark NN wins on test set!{Colors.END}")
    else:
        print(f"\n  {Colors.YELLOW}🤝 It's a tie!{Colors.END}")


def show_model_info(models):
    """Display model information."""
    print(f"\n{Colors.BOLD}Model Information:{Colors.END}")
    print(f"{'─' * 50}")
    
    print(f"\n{Colors.CYAN}Approach 1: ResNet50 Transfer Learning{Colors.END}")
    print(f"  • Input: 96×96 RGB images")
    print(f"  • Architecture: ResNet50 + custom head")
    print(f"  • Parameters: ~23M (base) + custom layers")
    if os.path.exists(RESNET_MODEL):
        size_mb = os.path.getsize(RESNET_MODEL) / (1024 * 1024)
        print(f"  • Model size: {size_mb:.1f} MB")
        print(f"  • Status: {Colors.GREEN}Loaded ✓{Colors.END}")
    else:
        print(f"  • Status: {Colors.RED}Not found ✗{Colors.END}")
    
    print(f"\n{Colors.CYAN}Approach 2: Landmark Neural Network{Colors.END}")
    print(f"  • Input: 63 features (21 landmarks × 3 coords)")
    print(f"  • Architecture: Dense NN (128→64→29)")
    print(f"  • Parameters: ~10K")
    if os.path.exists(LANDMARK_MODEL):
        size_kb = os.path.getsize(LANDMARK_MODEL) / 1024
        print(f"  • Model size: {size_kb:.1f} KB")
        print(f"  • Status: {Colors.GREEN}Loaded ✓{Colors.END}")
    else:
        print(f"  • Status: {Colors.RED}Not found ✗{Colors.END}")
    
    print(f"\n{'─' * 50}")


def interactive_mode(models):
    """Run interactive menu."""
    while True:
        print_menu()
        choice = input(f"{Colors.BOLD}Enter choice: {Colors.END}").strip().lower()
        
        if choice == '1':
            demo_test_set(models)
        elif choice == '2':
            path = input(f"\n{Colors.BOLD}Enter image path: {Colors.END}").strip()
            if os.path.exists(path):
                demo_single_image(path, models)
            else:
                print(f"{Colors.RED}File not found: {path}{Colors.END}")
        elif choice == '3':
            show_model_info(models)
        elif choice == 'q':
            print(f"\n{Colors.CYAN}Goodbye! 👋{Colors.END}\n")
            break
        else:
            print(f"{Colors.RED}Invalid choice. Try again.{Colors.END}")
        
        input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.END}")
        clear_screen()
        print_header()


def main():
    print_header()
    
    # Load models
    models = load_models()
    
    if not models:
        print(f"\n{Colors.RED}No models loaded. Please train the models first.{Colors.END}")
        return
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            demo_test_set(models)
        else:
            # Test specific image
            demo_single_image(sys.argv[1], models)
    else:
        # Interactive mode
        interactive_mode(models)
    
    print()


if __name__ == "__main__":
    main()
