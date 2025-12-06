# ASL Project Training & Execution Guide

## 🎯 Project Overview

This project implements **two approaches** for ASL alphabet recognition:
- **Approach 1**: ResNet50 Transfer Learning on raw images
- **Approach 2**: Neural Network on MediaPipe hand landmarks

Both approaches are trained on the [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) with 87,000+ images across 29 classes.

---

## ⚠️ Important Notes Before Starting

### Dataset Limitations
- The test set contains **only 29 images** (1 per class)
- This is insufficient for robust statistical evaluation
- Results should be interpreted with caution
- For production, create a proper held-out test set (split from training data)

### Hardware Requirements
- **Approach 1 (ResNet50)**: Requires GPU for reasonable training time
  - Expected: 2-4 hours with GPU, 10-20 hours with CPU
- **Approach 2 (Landmarks)**: CPU-friendly
  - Expected: 5-10 minutes on CPU

---

## 📋 Step-by-Step Execution Guide

### Step 1: Environment Setup

```bash
# Activate your conda environment
conda activate daml-asl

# Verify all dependencies are installed
pip install -r requirements.txt
```

**Expected output**: All packages should install without errors.

---

### Step 2: Data Preprocessing (Optional - Review Only)

**Notebook**: `01-data-preprocessing.ipynb`

This notebook demonstrates:
- Loading images using TensorFlow's `image_dataset_from_directory`
- Data augmentation (flips, rotation, zoom, translation, contrast)
- Normalization and optimization with caching/prefetching

**Action**: Run all cells to understand the preprocessing pipeline
**Time**: 2-3 minutes
**Output**: Visualizations of augmented training data

⚠️ **Note**: The actual training notebooks reload the data, so this is primarily educational.

---

### Step 3: MediaPipe Landmark Extraction

**Notebook**: `03-mediapipe-feature-extraction.ipynb`

This is **REQUIRED** for Approach 2.

**What it does**:
- Extracts 21 hand landmarks (63 features: x, y, z coordinates)
- Normalizes coordinates relative to wrist (translation-invariant)
- Uses parallel processing for speed (8 threads on most systems)
- Saves to `data/asl_landmarks_train.csv`

**Steps**:
1. Open `notebooks/03-mediapipe-feature-extraction.ipynb`
2. Run all cells sequentially
3. Wait for landmark extraction to complete

**Expected time**: 
- **With parallel processing**: 15-30 minutes
- **Output**: `../data/asl_landmarks_train.csv` (~86,000 rows × 64 columns)

**Success indicators**:
```
Processed 87000 images.
   Successful: ~85000 (97-98%)
   Failed: ~2000 (2-3%)
✓ Saved to ../data/asl_landmarks_train.csv
✓ CSV shape: (86912, 64)
✓ Classes: ['A', 'B', 'C', ..., 'Z', 'del', 'nothing', 'space']
```

**Common issues**:
- If detection rate is < 90%, check image quality
- If process is too slow, ensure parallel processing is enabled

---

### Step 4A: Train Approach 1 (ResNet50 Transfer Learning)

**Notebook**: `05-training-approach-1.ipynb`

**What it does**:
- Loads pre-trained ResNet50 (ImageNet weights)
- Freezes base model layers
- Adds custom classification head
- Two-phase training:
  - **Phase 1**: Train only new layers (10 epochs)
  - **Phase 2**: Fine-tune last ResNet layers (10 epochs)

**Steps**:
1. Open `notebooks/05-training-approach-1.ipynb`
2. Run cells 0-8 to start Phase 1 training
3. Wait for Phase 1 to complete (saves `best_asl_resnet50_phase1.h5`)
4. Run cells 9-12 for Phase 2 training
5. Wait for Phase 2 to complete (saves `best_asl_resnet50_phase2.h5`)

**Expected time**: 
- **With GPU**: 2-4 hours total
- **With CPU**: 10-20 hours (not recommended)

**Expected performance**:
- **Phase 1 Val Accuracy**: 85-95%
- **Phase 2 Val Accuracy**: 95-99%

**What to monitor**:
- Training should NOT overfit heavily (val_loss should stay close to train_loss)
- Early stopping will trigger if validation loss stops improving
- Learning rate reduction will trigger if plateauing

**Success indicators**:
```
Phase 1 (Frozen Base Model):
  Training Accuracy: 95.xx%
  Validation Accuracy: 93.xx%

Phase 2 (Fine-Tuned Model):
  Training Accuracy: 98.xx%
  Validation Accuracy: 97.xx%

✓ Model saved as 'best_asl_resnet50_phase2.h5'
```

**Saved files**:
- `best_asl_resnet50_phase1.h5` - Best model from Phase 1
- `best_asl_resnet50_phase2.h5` - Best model from Phase 2 (use this!)
- `asl_resnet50_final.h5` - Final model (may not be the best)

---

### Step 4B: Train Approach 2 (Landmark Neural Network)

**Notebook**: `06-training-approach-2-landmarks.ipynb`

**Prerequisites**: Must complete Step 3 (landmark extraction)

**What it does**:
- Loads landmarks from CSV
- Removes rows with failed detections (NaN values)
- Encodes labels and splits data (80/20 train/test)
- Normalizes features using StandardScaler
- Trains a simple neural network (128→64→29 neurons)
- Uses early stopping (patience=5)

**Steps**:
1. Ensure `data/asl_landmarks_train.csv` exists
2. Open `notebooks/06-training-approach-2-landmarks.ipynb`
3. Run all cells sequentially

**Expected time**: 5-10 minutes

**Expected performance**:
- **Test Accuracy**: 85-95%
- Training typically converges in 15-30 epochs

**What to monitor**:
- Training/validation accuracy curves should both increase
- Training should stop early (before 50 epochs) if converged

**Success indicators**:
```
Samples: 85000 | Features: 63 | Classes: 29
Train: 68000 | Test: 17000

Test Accuracy: 92.xx%

✓ Model saved!
```

**Saved files**:
- `data/nn_landmark_model.keras` - Trained neural network
- `data/label_encoder.joblib` - Label encoder for predictions
- `data/scaler.joblib` - Feature scaler

---

### Step 5: Model Comparison & Evaluation

**Notebook**: `07-model-comparison-evaluation.ipynb`

**Prerequisites**: Complete both Step 4A and Step 4B

**What it does**:
- Loads both trained models
- Evaluates on test set (29 images)
- Compares performance side-by-side
- Provides insights and recommendations

**Steps**:
1. Open `notebooks/07-model-comparison-evaluation.ipynb`
2. Run all cells sequentially

**Expected time**: 1-2 minutes

**Expected output**:
```
MODEL COMPARISON SUMMARY
============================================================

Approach 1 - ResNet50 Transfer Learning:
  Test Accuracy: 96.xx%
  Input: 96x96 RGB images
  Model Size: ~23M parameters

Approach 2 - Landmark Neural Network:
  Test Accuracy: 93.xx%
  Input: 63 landmark features
  Model Size: ~10K parameters

✓ Winner: [Model with higher accuracy]
```

**Note**: With only 29 test samples, differences < 10% are not statistically significant.

---

## 🎓 For Your Final Presentation

Based on the project spec, your presentation should address:

### 1. Accuracy Across Different Conditions
- **What to show**: Performance metrics from both approaches
- **Where to find**: Notebook 07 comparison results
- **Key insight**: Discuss which approach is more robust and why

### 2. Important Hand Landmark Features
- **What to show**: Feature importance analysis for Approach 2
- **Where to add**: You could extend notebook 06 to show which landmarks contribute most
- **Suggestion**: Visualize the 21 hand landmarks and highlight the most important ones

### 3. Scaling to Words/Phrases
- **What to discuss**: 
  - Current system: static images → letters
  - Next step: video sequences → words
  - Technical approach: LSTM/Transformer on temporal sequences
  - Challenges: co-articulation, timing, grammar differences

### 4. Real-World Integration
- **Classroom scenario**: 
  - Approach 2 (lightweight) on tablets/phones
  - Real-time feedback for students learning ASL
  - Low latency, works offline
- **Assistive device scenario**:
  - Wearable camera with edge computing
  - Hybrid model for high accuracy
  - Privacy-preserving (local processing)

---

## 📊 Expected Results Summary

| Metric | Approach 1 (ResNet50) | Approach 2 (Landmarks) |
|--------|----------------------|------------------------|
| **Validation Accuracy** | 95-99% | 85-95% |
| **Test Accuracy** | 90-100% | 85-95% |
| **Training Time** | 2-4 hours (GPU) | 5-10 minutes |
| **Model Size** | ~100 MB | ~1 MB |
| **Inference Speed** | Slow (needs GPU) | Fast (CPU-friendly) |
| **Robustness** | High | Medium (depends on detection) |

---

## 🐛 Troubleshooting

### Issue: "No module named X"
**Solution**: 
```bash
pip install -r requirements.txt
```

### Issue: Notebook 03 is very slow
**Solution**: Ensure parallel processing is enabled (should show "Using 8 threads")

### Issue: Landmark CSV not found in Notebook 06
**Solution**: Run Notebook 03 first to generate the CSV

### Issue: Model file not found in Notebook 07
**Solution**: Train both models (Notebooks 05 and 06) first

### Issue: CUDA/GPU errors
**Solution**: 
- For ResNet50: Ensure TensorFlow GPU is properly installed
- Alternative: Train on CPU (will take longer)

### Issue: Training accuracy high but validation accuracy low
**Solution**: 
- This is overfitting
- Increase dropout rates
- Add more data augmentation
- Reduce model complexity

---

## 🎯 Quick Start (If You Just Want Results)

If you're short on time and want to see results quickly:

1. **Generate landmarks** (Notebook 03) - 30 minutes
2. **Train Landmark NN** (Notebook 06) - 10 minutes
3. **Evaluate** (Notebook 07) - 2 minutes

**Total time**: ~45 minutes for working results

For best results, also train ResNet50 (adds 2-4 hours with GPU).

---

## 📝 Next Steps & Future Work

1. **Improve Test Set**:
   - Create proper held-out test set from training data
   - Use stratified split to ensure all classes represented
   - Aim for at least 500-1000 test samples

2. **Real-Time Webcam Demo**:
   - Capture live video
   - Process frames in real-time
   - Display predictions with confidence

3. **Expand Scope**:
   - Add temporal models (LSTM) for continuous signing
   - Include ASL words beyond alphabet
   - Account for sentence structure and grammar

4. **Production Deployment**:
   - Convert to TensorFlow Lite for mobile
   - Create REST API for web integration
   - Build user-friendly interface

---

## 📚 Additional Resources

- **MediaPipe Hand Landmarks**: https://google.github.io/mediapipe/solutions/hands.html
- **ResNet Paper**: https://arxiv.org/abs/1512.03385
- **Transfer Learning Guide**: https://www.tensorflow.org/tutorials/images/transfer_learning
- **ASL Linguistics**: https://www.lifeprint.com/

---

## ✅ Checklist for Completion

- [ ] Environment setup complete
- [ ] Reviewed data preprocessing (Notebook 01)
- [ ] Generated landmarks CSV (Notebook 03)
- [ ] Trained ResNet50 model (Notebook 05)
- [ ] Trained Landmark NN (Notebook 06)
- [ ] Compared both models (Notebook 07)
- [ ] Prepared presentation materials
- [ ] Tested key insights and findings

---

**Good luck with your project! 🚀**

For questions or issues, refer to this guide or check the notebook documentation.

