import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
import random
from pathlib import Path

# ============================
# CONFIGURATION
# ============================
IMG_SIZE = (768, 1024)
MODEL_PATH = r"C:\Users\peter\Desktop\Courses\Machine Learning\project\checkpoints_stable\epoch-025.keras"
HISTORY_PATH = r"C:\Users\peter\Desktop\Courses\Machine Learning\project\training_history_stable.pkl"
IMAGE_ROOT = r"C:\Users\peter\Desktop\Courses\Machine Learning\project\RaindropsOnWindshield\images"
MASK_ROOT = r"C:\Users\peter\Desktop\Courses\Machine Learning\project\RaindropsOnWindshield\masks"

# Output directory for plots and results
OUTPUT_DIR = r"C:\Users\peter\Desktop\Courses\Machine Learning\project\evaluation_results_corrected"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading model and history...")

# ============================
# LOAD MODEL AND HISTORY
# ============================
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        'combined_loss': combined_loss,
        'dice_coefficient': dice_coefficient
    }
)

with open(HISTORY_PATH, "rb") as f:
    history = pickle.load(f)

print(f"✅ Model loaded from: {MODEL_PATH}")
print(f"✅ Training history loaded: {len(history['loss'])} epochs")

# ============================
# PLOT 1A: LOSS CURVES (SEPARATE)
# ============================
print("\n📊 Creating loss curves...")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(history['loss'], label='Training Loss', linewidth=2)
ax.plot(history['val_loss'], label='Validation Loss', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Loss vs Epoch', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curves.png'), dpi=300, bbox_inches='tight')
print(f"✅ Saved: loss_curves.png")
plt.close()

# ============================
# PLOT 1B: DICE AND ACCURACY CURVES
# ============================
print("\n📊 Creating dice and accuracy curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training History', fontsize=16, fontweight='bold')

# Dice Coefficient
axes[0].plot(history['dice_coefficient'], label='Training Dice', linewidth=2)
axes[0].plot(history['val_dice_coefficient'], label='Validation Dice', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Dice Coefficient')
axes[0].set_title('Dice Coefficient vs Epoch')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
axes[1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy vs Epoch')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'dice_accuracy_curves.png'), dpi=300, bbox_inches='tight')
print(f"✅ Saved: dice_accuracy_curves.png")
plt.close()
# ============================
# PLOT 2: FINAL METRICS SUMMARY
# ============================
print("\n📊 Creating metrics summary...")

best_epoch = np.argmax(history['val_dice_coefficient'])
best_val_dice = history['val_dice_coefficient'][best_epoch]
final_train_dice = history['dice_coefficient'][-1]
final_val_dice = history['val_dice_coefficient'][-1]

fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['Best Val Dice', 'Final Train Dice', 'Final Val Dice']
values = [best_val_dice, final_train_dice, final_val_dice]
colors = ['#2ecc71', '#3498db', '#e74c3c']

bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Dice Coefficient', fontsize=12)
ax.set_title('Model Performance Summary', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1])
ax.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.4f}\n({val*100:.2f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'metrics_summary.png'), dpi=300, bbox_inches='tight')
print(f"✅ Saved: metrics_summary.png")
plt.close()

# ============================
# GET ONE IMAGE FROM EACH SUBFOLDER
# ============================
print("\n🔍 Finding sample images from each subfolder...")

subfolders = {}
for root, _, files in os.walk(IMAGE_ROOT):
    if files:
        folder_name = os.path.basename(root)
        if folder_name and folder_name != os.path.basename(IMAGE_ROOT):
            img_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if img_files:
                # Pick a random image from this folder
                sample_img = random.choice(img_files)
                img_path = os.path.join(root, sample_img)
                rel_path = os.path.relpath(img_path, IMAGE_ROOT)
                mask_path = os.path.join(MASK_ROOT, rel_path)
                
                if os.path.exists(mask_path):
                    subfolders[folder_name] = {
                        'image': img_path,
                        'mask': mask_path
                    }

print(f"Found {len(subfolders)} subfolders with valid image/mask pairs")

# ============================
# EVALUATE EACH SUBFOLDER
# ============================
print("\n📈 Evaluating samples from each subfolder...")

results_with_droplets = []
results_no_droplets = []

for folder_name, paths in sorted(subfolders.items()):
    try:
        # Load and preprocess
        img = load_img(paths['image'], target_size=IMG_SIZE)
        mask = load_img(paths['mask'], target_size=IMG_SIZE, color_mode='grayscale')
        
        img_array = img_to_array(img) / 255.0
        mask_array = img_to_array(mask) / 255.0
        mask_binary = (mask_array > 0.5).astype(np.float32)
        
        # Check if mask has droplets
        has_droplets = np.sum(mask_binary) > 0
        
        # Predict
        pred = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]
        pred_binary = (pred > 0.5).astype(np.float32)
        
        # Calculate metrics
        intersection = np.sum(pred_binary * mask_binary)
        union = np.sum(pred_binary) + np.sum(mask_binary)
        
        # False positives and negatives
        fp = np.sum((pred_binary == 1) & (mask_binary == 0))
        fn = np.sum((pred_binary == 0) & (mask_binary == 1))
        tp = np.sum((pred_binary == 1) & (mask_binary == 1))
        tn = np.sum((pred_binary == 0) & (mask_binary == 0))
        
        total_pixels = mask_binary.size
        
        result = {
            'folder': folder_name,
            'has_droplets': has_droplets,
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'tn': int(tn),
            'total_pixels': total_pixels,
            'image': img_array,
            'mask': mask_binary,
            'pred': pred
        }
        
        if has_droplets:
            # Standard metrics for droplet detection
            dice = (2. * intersection + 1e-6) / (union + 1e-6)
            iou = intersection / (np.sum(pred_binary) + np.sum(mask_binary) - intersection + 1e-6)
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
            accuracy = (tp + tn) / total_pixels
            
            result.update({
                'dice': dice,
                'iou': iou,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            results_with_droplets.append(result)
            print(f"  {folder_name} (WITH droplets): Dice={dice:.4f}, IoU={iou:.4f}, F1={f1:.4f}")
        else:
            # Specificity metrics for clean images
            specificity = tn / (tn + fp + 1e-6)
            false_positive_rate = fp / total_pixels
            
            result.update({
                'specificity': specificity,
                'false_positive_rate': false_positive_rate
            })
            
            results_no_droplets.append(result)
            print(f"  {folder_name} (NO droplets): Specificity={specificity:.4f}, FP Rate={false_positive_rate:.6f}")
        
    except Exception as e:
        print(f"  ⚠ Error processing {folder_name}: {e}")

# ============================
# PLOT 3: SEPARATE METRICS FOR DROPLET VS NO-DROPLET
# ============================
print("\n📊 Creating separated performance plots...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: Dice/IoU/F1 for folders WITH droplets
if results_with_droplets:
    ax1 = fig.add_subplot(gs[0, :])
    folders = [r['folder'] for r in results_with_droplets]
    dice_scores = [r['dice'] for r in results_with_droplets]
    iou_scores = [r['iou'] for r in results_with_droplets]
    f1_scores = [r['f1'] for r in results_with_droplets]
    
    x = np.arange(len(folders))
    width = 0.25
    
    ax1.bar(x - width, dice_scores, width, label='Dice', alpha=0.8, color='#3498db')
    ax1.bar(x, iou_scores, width, label='IoU', alpha=0.8, color='#2ecc71')
    ax1.bar(x + width, f1_scores, width, label='F1', alpha=0.8, color='#e74c3c')
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Raindrop Detection Performance (Folders WITH Droplets)', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(folders, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_ylim([0, 1])

# Plot 2: Precision/Recall for folders WITH droplets
if results_with_droplets:
    ax2 = fig.add_subplot(gs[1, :])
    precisions = [r['precision'] for r in results_with_droplets]
    recalls = [r['recall'] for r in results_with_droplets]
    
    ax2.bar(x - width/2, precisions, width, label='Precision', alpha=0.8, color='green')
    ax2.bar(x + width/2, recalls, width, label='Recall', alpha=0.8, color='orange')
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Precision and Recall (Folders WITH Droplets)', fontweight='bold', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(folders, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_ylim([0, 1])

# Plot 3: False Positive Rate for folders WITHOUT droplets
if results_no_droplets:
    ax3 = fig.add_subplot(gs[2, 0])
    no_drop_folders = [r['folder'] for r in results_no_droplets]
    fp_rates = [r['false_positive_rate'] * 100 for r in results_no_droplets]  # Convert to percentage
    
    bars = ax3.bar(no_drop_folders, fp_rates, alpha=0.8, color='#e74c3c', edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('False Positive Rate (%)', fontsize=12)
    ax3.set_title('False Positives on Clean Images (NO Droplets)', fontweight='bold', fontsize=12)
    ax3.set_xticklabels(no_drop_folders, rotation=45, ha='right')
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, fp_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}%',
                ha='center', va='bottom', fontsize=9)

# Plot 4: Specificity for folders WITHOUT droplets
if results_no_droplets:
    ax4 = fig.add_subplot(gs[2, 1])
    specificities = [r['specificity'] * 100 for r in results_no_droplets]
    
    bars = ax4.bar(no_drop_folders, specificities, alpha=0.8, color='#2ecc71', edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Specificity (%)', fontsize=12)
    ax4.set_title('Specificity on Clean Images (NO Droplets)', fontweight='bold', fontsize=12)
    ax4.set_xticklabels(no_drop_folders, rotation=45, ha='right')
    ax4.grid(True, axis='y', alpha=0.3)
    ax4.set_ylim([99, 100])
    
    # Add value labels
    for bar, val in zip(bars, specificities):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%',
                ha='center', va='bottom', fontsize=9)

plt.savefig(os.path.join(OUTPUT_DIR, 'separated_performance_metrics.png'), dpi=300, bbox_inches='tight')
print(f"✅ Saved: separated_performance_metrics.png")
plt.close()
# ============================
# PLOT 3.5: DICE SCORES BAR CHART FOR EACH FOLDER
# ============================
if results_with_droplets:
    print("\n📊 Creating Dice scores bar chart...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    folders = [r['folder'] for r in results_with_droplets]
    dice_scores = [r['dice'] for r in results_with_droplets]
    
    # Create color gradient based on dice scores
    colors = plt.cm.RdYlGn([d for d in dice_scores])
    
    bars = ax.bar(folders, dice_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Dice Coefficient', fontsize=12)
    ax.set_xlabel('Folder', fontsize=12)
    ax.set_title('Dice Coefficient by Folder (WITH Droplets)', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, axis='y', alpha=0.3)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, val in zip(bars, dice_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add horizontal line for average
    avg_dice = np.mean(dice_scores)
    ax.axhline(y=avg_dice, color='red', linestyle='--', linewidth=2, 
               label=f'Average: {avg_dice:.3f}')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dice_scores_by_folder.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: dice_scores_by_folder.png")
    plt.close()

# ============================
# PLOT 4: SAMPLE PREDICTIONS - WITH DROPLETS
# ============================
if results_with_droplets:
    print("\n📊 Creating sample predictions (with droplets)...")
    
    n_samples = min(len(results_with_droplets), 12)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(results_with_droplets[:n_samples]):
        # Input image
        axes[idx, 0].imshow(result['image'])
        axes[idx, 0].set_title(f"{result['folder']}\nInput Image", fontsize=18)
        axes[idx, 0].axis('off')
        
        # Ground truth mask
        axes[idx, 1].imshow(result['mask'].squeeze(), cmap='gray')
        axes[idx, 1].set_title(f"Training Mask\n(Actual Raindrops)", fontsize=18)
        axes[idx, 1].axis('off')
        
        # Predicted mask
        axes[idx, 2].imshow(result['pred'].squeeze(), cmap='gray')
        axes[idx, 2].set_title(f"Prediction\nDice: {result['dice']:.3f}", fontsize=18)
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_predictions_with_droplets.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: sample_predictions_with_droplets.png")
    plt.close()

# ============================
# PLOT 5: SAMPLE PREDICTIONS - NO DROPLETS
# ============================
if results_no_droplets:
    print("\n📊 Creating sample predictions (no droplets)...")
    
    n_samples = min(len(results_no_droplets), 6)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5*n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(results_no_droplets[:n_samples]):
        # Input image
        axes[idx, 0].imshow(result['image'])
        axes[idx, 0].set_title(f"{result['folder']}\nInput Image (No Droplets)", fontsize=18)
        axes[idx, 0].axis('off')
        
        # Ground truth mask
        axes[idx, 1].imshow(result['mask'].squeeze(), cmap='gray')
        axes[idx, 1].set_title(f"Training Mask\n(Clean - No Droplets)", fontsize=18)
        axes[idx, 1].axis('off')
        
        # Predicted mask
        axes[idx, 2].imshow(result['pred'].squeeze(), cmap='gray')
        fp_rate = result['false_positive_rate'] * 100
        axes[idx, 2].set_title(f"Prediction\nFP Rate: {fp_rate:.3f}%", fontsize=18)
        axes[idx, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_predictions_no_droplets.png'), dpi=300, bbox_inches='tight')
    print(f"✅ Saved: sample_predictions_no_droplets.png")
    plt.close()

# ============================
# SAVE DETAILED RESULTS TO TEXT FILE
# ============================
print("\n📝 Saving detailed results to text file...")

with open(os.path.join(OUTPUT_DIR, 'evaluation_report_corrected.txt'), 'w') as f:
    f.write("="*80 + "\n")
    f.write("RAINDROP DETECTION MODEL - CORRECTED EVALUATION REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Total Epochs Trained: {len(history['loss'])}\n")
    f.write(f"Best Validation Dice: {best_val_dice:.4f} (Epoch {best_epoch + 1})\n")
    f.write(f"Final Training Dice: {final_train_dice:.4f}\n")
    f.write(f"Final Validation Dice: {final_val_dice:.4f}\n")
    f.write(f"Final Validation Accuracy: {history['val_accuracy'][-1]:.4f} ({history['val_accuracy'][-1]*100:.2f}%)\n\n")
    
    f.write("="*80 + "\n")
    f.write("RAINDROP DETECTION PERFORMANCE (Folders WITH Droplets)\n")
    f.write("="*80 + "\n\n")
    
    for result in results_with_droplets:
        f.write(f"Folder: {result['folder']}\n")
        f.write(f"  Dice Coefficient:  {result['dice']:.4f} ({result['dice']*100:.2f}%)\n")
        f.write(f"  IoU (Jaccard):     {result['iou']:.4f} ({result['iou']*100:.2f}%)\n")
        f.write(f"  Pixel Accuracy:    {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)\n")
        f.write(f"  Precision:         {result['precision']:.4f}\n")
        f.write(f"  Recall:            {result['recall']:.4f}\n")
        f.write(f"  F1 Score:          {result['f1']:.4f}\n")
        f.write(f"  True Positives:    {result['tp']:,} pixels\n")
        f.write(f"  False Positives:   {result['fp']:,} pixels\n")
        f.write(f"  False Negatives:   {result['fn']:,} pixels\n")
        f.write(f"  True Negatives:    {result['tn']:,} pixels\n")
        f.write("\n")
    
    f.write("="*80 + "\n")
    f.write("FALSE POSITIVE ANALYSIS (Folders WITHOUT Droplets)\n")
    f.write("="*80 + "\n\n")
    
    for result in results_no_droplets:
        f.write(f"Folder: {result['folder']}\n")
        f.write(f"  Specificity:           {result['specificity']:.6f} ({result['specificity']*100:.4f}%)\n")
        f.write(f"  False Positive Rate:   {result['false_positive_rate']:.6f} ({result['false_positive_rate']*100:.4f}%)\n")
        f.write(f"  False Positives:       {result['fp']:,} pixels\n")
        f.write(f"  True Negatives:        {result['tn']:,} pixels\n")
        f.write(f"  Total Pixels:          {result['total_pixels']:,} pixels\n")
        f.write("\n")
    
    f.write("="*80 + "\n")
    f.write("AGGREGATE STATISTICS\n")
    f.write("="*80 + "\n\n")
    
    if results_with_droplets:
        avg_dice = np.mean([r['dice'] for r in results_with_droplets])
        avg_iou = np.mean([r['iou'] for r in results_with_droplets])
        avg_f1 = np.mean([r['f1'] for r in results_with_droplets])
        avg_precision = np.mean([r['precision'] for r in results_with_droplets])
        avg_recall = np.mean([r['recall'] for r in results_with_droplets])
        
        f.write("RAINDROP DETECTION (Folders with droplets):\n")
        f.write(f"  Average Dice Coefficient: {avg_dice:.4f} ({avg_dice*100:.2f}%)\n")
        f.write(f"  Average IoU:              {avg_iou:.4f} ({avg_iou*100:.2f}%)\n")
        f.write(f"  Average F1 Score:         {avg_f1:.4f} ({avg_f1*100:.2f}%)\n")
        f.write(f"  Average Precision:        {avg_precision:.4f} ({avg_precision*100:.2f}%)\n")
        f.write(f"  Average Recall:           {avg_recall:.4f} ({avg_recall*100:.2f}%)\n")
        f.write(f"  Std Dev Dice:             {np.std([r['dice'] for r in results_with_droplets]):.4f}\n")
        f.write(f"  Min Dice:                 {np.min([r['dice'] for r in results_with_droplets]):.4f}\n")
        f.write(f"  Max Dice:                 {np.max([r['dice'] for r in results_with_droplets]):.4f}\n\n")
    
    if results_no_droplets:
        avg_specificity = np.mean([r['specificity'] for r in results_no_droplets])
        avg_fp_rate = np.mean([r['false_positive_rate'] for r in results_no_droplets])
        
        f.write("FALSE POSITIVE CONTROL (Folders without droplets):\n")
        f.write(f"  Average Specificity:      {avg_specificity:.6f} ({avg_specificity*100:.4f}%)\n")
        f.write(f"  Average FP Rate:          {avg_fp_rate:.6f} ({avg_fp_rate*100:.4f}%)\n")
        f.write(f"  Total False Positives:    {sum(r['fp'] for r in results_no_droplets):,} pixels\n")
        f.write(f"  Total Pixels Analyzed:    {sum(r['total_pixels'] for r in results_no_droplets):,} pixels\n")

print(f"✅ Saved: evaluation_report_corrected.txt")

# ============================
# SUMMARY
# ============================
print("\n" + "="*80)
print("CORRECTED EVALUATION COMPLETE!")
print("="*80)
print(f"\n📁 All results saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  1. training_curves.png                      - Training metrics over time")
print("  2. metrics_summary.png                      - Overall performance summary")
print("  3. separated_performance_metrics.png        - Separate analysis for droplet/no-droplet")
print("  4. sample_predictions_with_droplets.png     - Visual examples (with droplets)")
print("  5. sample_predictions_no_droplets.png       - Visual examples (no droplets)")
print("  6. evaluation_report_corrected.txt          - Detailed numerical results")
print("\n" + "="*80)
print(f"\n🎯 FINAL MODEL PERFORMANCE:")
print(f"   Best Validation Dice:        {best_val_dice:.4f} ({best_val_dice*100:.2f}%)")
if results_with_droplets:
    print(f"   Avg Dice (with droplets):    {avg_dice:.4f} ({avg_dice*100:.2f}%)")
if results_no_droplets:
    print(f"   Avg Specificity (no drops):  {avg_specificity:.6f} ({avg_specificity*100:.4f}%)")
    print(f"   Avg FP Rate (no drops):      {avg_fp_rate:.6f} ({avg_fp_rate*100:.4f}%)")
print("="*80 + "\n")

# ============================
# PLOT 6: YOUR OWN PHOTO TEST
# ============================
print("\n📸 Testing on your own photo...")

CUSTOM_PHOTO_DIR = r"C:\Users\peter\Desktop\Courses\Machine Learning\project\my_photos"

if os.path.exists(CUSTOM_PHOTO_DIR):
    custom_images = [f for f in os.listdir(CUSTOM_PHOTO_DIR) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if custom_images:
        print(f"Found {len(custom_images)} custom image(s)")
        
        # Create figure based on number of images
        n_custom = len(custom_images)
        fig, axes = plt.subplots(n_custom, 2, figsize=(12, 5*n_custom))
        if n_custom == 1:
            axes = axes.reshape(1, -1)
        
        for idx, img_name in enumerate(custom_images):
            img_path = os.path.join(CUSTOM_PHOTO_DIR, img_name)
            
            # Load and preprocess
            img = load_img(img_path, target_size=IMG_SIZE)
            img_array = img_to_array(img) / 255.0
            
            # Predict
            pred = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]
            pred_binary = (pred > 0.5).astype(np.float32)
            
            # Count detected pixels
            droplet_pixels = np.sum(pred_binary)
            total_pixels = pred_binary.size
            droplet_percentage = (droplet_pixels / total_pixels) * 100
            
            # Original image
            axes[idx, 0].imshow(img_array)
            axes[idx, 0].set_title(f"Test Photo", fontsize=12)
            axes[idx, 0].axis('off')
            
            # Prediction
            axes[idx, 1].imshow(pred.squeeze(), cmap='gray')
            axes[idx, 1].set_title(f"Detected Raindrops\n{droplet_pixels:,} pixels ({droplet_percentage:.2f}%)", fontsize=12)
            axes[idx, 1].axis('off')
            
            print(f"  {img_name}: Detected {droplet_pixels:,} raindrop pixels ({droplet_percentage:.2f}%)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'custom_photo_predictions.png'), dpi=300, bbox_inches='tight')
        print(f"✅ Saved: custom_photo_predictions.png")
        plt.close()
        
        # Also save individual prediction masks
        for img_name in custom_images:
            img_path = os.path.join(CUSTOM_PHOTO_DIR, img_name)
            img = load_img(img_path, target_size=IMG_SIZE)
            img_array = img_to_array(img) / 255.0
            pred = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0]
            
            # Save the raw prediction mask
            mask_name = os.path.splitext(img_name)[0] + '_mask.png'
            plt.imsave(os.path.join(OUTPUT_DIR, mask_name), pred.squeeze(), cmap='gray')
            print(f"✅ Saved mask: {mask_name}")
    else:
        print(f"⚠ No images found in {CUSTOM_PHOTO_DIR}")
        print("  Add .jpg, .jpeg, or .png files to test your own photos")
else:
    print(f"⚠ Custom photo directory not found: {CUSTOM_PHOTO_DIR}")
    print("  Create this folder and add your photos to test them")
